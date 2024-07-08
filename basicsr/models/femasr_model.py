from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy
from .cal_ssim import SSIM

import pyiqa

import numpy as np
import cv2
import math


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.
    Args:
        sample (dict): sample
        size (tuple): image size
    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
            self,
            width,
            height,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=1,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample


from torch import nn


@MODEL_REGISTRY.register()
class FeMaSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.ssim = SSIM().cuda()
        self.output_val_wo_prompt = None
        self.output_val_w_prompt = None
        ############################## depth estimation network #########################
        self.model_type = opt['datasets']['train']['model_type']
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.cuda()
        self.midas.eval()
        self.l1 = nn.L1Loss().cuda()
        ##############################

        # define metric functions
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # load pre-trained HQ ckpt, frozen decoder and codebook
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            hq_opt = self.opt['network_g'].copy()
            hq_opt['LQ_stage'] = False
            self.net_hq = build_network(hq_opt)
            self.net_hq = self.model_to_device(self.net_hq)
            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            self.load_network(self.net_g, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        # print('#########################################################################',load_path)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        # print(load_path)
        if load_path is not None:
            logger.info(f'Loading net_d from {load_path}')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.model_to_device(self.cri_perceptual)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.depth_transform_gt = data['depth_transform_gt'].to(self.device)
        self.depth_transform_lq = data['depth_transform_lq'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        # self.depth1_reconstruction = None
        # self.depth2_reconstruction = None
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()
        h, c, h, w = self.lq.size()
        if self.LQ_stage:
            with torch.no_grad():
                depth1 = self.midas(self.depth_transform_lq)
                depth2 = self.midas(self.depth_transform_gt)
                depth1 = (depth1 * 255 / torch.max(depth1))
                depth2 = (depth2 * 255 / torch.max(depth2))
                self.depth1 = torch.nn.functional.interpolate(depth1.unsqueeze(1), size=(h, w), mode="bicubic",
                                                              align_corners=False, ) / 255.0
                self.depth2 = torch.nn.functional.interpolate(depth2.unsqueeze(1), size=(h, w), mode="bicubic",
                                                              align_corners=False, ) / 255.0
                self.gt_rec, _, _, gt_indices, _, depth_quant_1, depth_quant_2, self.depth1, self.depth2 = self.net_hq(
                    input=self.gt, second_img=self.lq, depth1=self.depth1, depth2=self.depth2)
            # self.output_woprompt, l_codebook_woprompt, l_codebook_second_woprompt, _, _, _, _, _, _ = self.net_g(
            #     input=self.lq,gt_indices=gt_indices)
            self.output, l_codebook, l_codebook_second, _, _, _, _, _, _ = self.net_g(input=self.lq, second_img=self.gt,
                                                                                      depth_quant_1=depth_quant_1,
                                                                                      depth_quant_2=depth_quant_2,
                                                                                      gt_indices=gt_indices)
        else:
            self.output, l_codebook, _, _, _, _, _, _, _ = self.net_g(self.gt)

        l_g_total = 0
        loss_dict = OrderedDict()
        # ===================================================
        # codebook loss
        if train_opt.get('codebook_opt', None) and l_codebook != 0:
            l_codebook *= train_opt['codebook_opt']['loss_weight']
            l_g_total += l_codebook.mean()
            loss_dict['l_codebook'] = l_codebook.mean()

        # codebook second loss
        if train_opt.get('codebook_second_opt', None):
            l_codebook_second *= train_opt['codebook_second_opt']['loss_weight']
            l_g_total += l_codebook_second.mean()
            loss_dict['l_codebook_second'] = l_codebook_second.mean()
        # ###################pixel loss  pixel_woprompt_opt
        # if train_opt.get('pixel_woprompt_opt', None):
        #     l_pix_woprompt = train_opt['pixel_woprompt_opt']['loss_weight'] * self.l1(self.output_woprompt, self.gt)
        #     l_g_total += l_pix_woprompt
        #     loss_dict['l_pixel_woprompt'] = l_pix_woprompt
        #
        # if train_opt.get('codebook_woprompt_opt', None):
        #     l_codebook_woprompt = train_opt['codebook_woprompt_opt']['loss_weight'] * l_codebook_second_woprompt
        #     l_g_total += l_codebook_woprompt
        #     loss_dict['l_codebook_woprompt'] = l_codebook_woprompt
        #
        # if train_opt.get('ssim_woprompt_opt', None):
        #     l_ssim_woprompt = train_opt['ssim_woprompt_opt']['loss_weight'] \
        #                           * (1 - self.ssim(self.output_woprompt, self.gt))
        #     l_g_total += l_ssim_woprompt
        #     loss_dict['l_ssim_woprompt'] = l_ssim_woprompt
        #
        # if train_opt.get('perceptual_woprompt_opt', None):
        #     l_percep_woprompt, l_style_woprompt = self.cri_perceptual(self.output_woprompt, self.gt)
        #     if l_percep_woprompt is not None:
        #         l_g_total += l_percep_woprompt.mean()
        #         loss_dict['l_percep_woprompt'] = l_percep_woprompt.mean()
        #     if l_style_woprompt is not None:
        #         l_g_total += l_style_woprompt
        #         loss_dict['l_style_woprompt'] = l_style_woprompt
        #######################################################################

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        if train_opt.get('ssim_loss_opt', None):
            l_ssim = train_opt['ssim_loss_opt']['loss_weight'] * (1 - self.ssim(self.output, self.gt))
            l_g_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style

        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['train'].get('fixed_disc', False)
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def transform_module(self):
        import cv2
        from torchvision.transforms import Compose
        # from transforms_module import Resize, NormalizeImage, PrepareForNet
        model_type = self.model_type
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            dpt_transform = Compose(
                [
                    lambda img: {"image": img / 255.0},
                    Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    PrepareForNet(),
                    # lambda sample: torch.from_numpy(sample["image"]),
                    lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
                ]
            )
            return dpt_transform
        elif model_type == "MiDaS_small":
            small_transform = Compose(
                [
                    lambda img: {"image": img / 255.0},
                    Resize(
                        256,
                        256,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                    # lambda sample: torch.from_numpy(sample["image"]),
                    lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
                ]
            )
            return small_transform
        else:
            default_transform = Compose(
                [
                    lambda img: {"image": img / 255.0},
                    Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                    # lambda sample: torch.from_numpy(sample["image"]),
                    lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
                ]
            )
            return default_transform

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        _, _, h, w = lq_input.shape
        self.output_val_wo_prompt = net_g.test(input=lq_input)
        depth1 = self.midas(self.depth_transform_lq)
        transform = self.transform_module()
        output_val_wo_prompt = transform(self.output_val_wo_prompt.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)
        depth2 = self.midas(output_val_wo_prompt.cuda())

        depth1 = (depth1 * 255 / torch.max(depth1))

        depth2 = (depth2 * 255 / torch.max(depth2))

        self.depth1 = torch.nn.functional.interpolate(
            depth1.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ) / 255.0

        self.depth2 = torch.nn.functional.interpolate(
            depth2.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ) / 255.0

        ######################################################
        _, _, _, _, _, depth_quant_1, depth_quant_2, _, _ = self.net_hq(input=lq_input, second_img=lq_input,
                                                                        depth1=self.depth1, depth2=self.depth2)
        self.output = net_g.test(input=lq_input, depth_quant_1=depth_quant_1, depth_quant_2=depth_quant_2)
        self.output_val_w_prompt = self.output
        # else:
        #     self.output = net_g.test_tile(lq_input)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            sr_img = tensor2img(self.output)

            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
                    self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]

        if self.depth1 is not None:
            out_dict['depth1'] = self.depth1.detach().cpu()[:vis_samples]
        if self.depth2 is not None:
            out_dict['depth2'] = self.depth2.detach().cpu()[:vis_samples]

        # if self.depth1_reconstruction is not None:
        #     out_dict['depth1_reconstruction'] = self.depth1_reconstruction.detach().cpu()[:vis_samples]
        #
        # if self.depth2_reconstruction is not None:
        #     out_dict['depth2_reconstruction'] = self.depth2_reconstruction.detach().cpu()[:vis_samples]

        if not self.LQ_stage:
            out_dict['result'] = self.output.detach().cpu()[:vis_samples]

        # self.output_val_wo_prompt = None
        # self.output_val_w_prompt = None
        if self.LQ_stage:
            if self.output_val_wo_prompt is not None:
                out_dict['output_val_wo_prompt'] = self.output_val_wo_prompt.detach().cpu()[:vis_samples]

            if self.output_val_w_prompt is not None:
                out_dict['output_val_w_prompt'] = self.output_val_w_prompt.detach().cpu()[:vis_samples]

            # if self.output is not None:
        if self.LQ_stage:
            out_dict['output_w_prompt'] = self.output.detach().cpu()[:vis_samples]

        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter, epoch)
        self.save_network(self.net_d, 'net_d', current_iter, epoch)
        self.save_training_state(epoch, current_iter)
