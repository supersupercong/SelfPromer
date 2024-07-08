import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils.download_util import load_file_from_url

pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}


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

def transform_module(model_type):
    import cv2
    from torchvision.transforms import Compose
    # from transforms_module import Resize, NormalizeImage, PrepareForNet
    model_type = model_type
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main():
    """Inference demo for FeMaSR
    """
    print('---------------')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/data_8T1/wangcong/dataset/Teton_0-0', help='Input image or folder')
    parser.add_argument('-w_dehaze', '--weight_dehaze', type=str, default='./experiments/014_FeMaSR_LQ_stage/models/net_g_300000.pth', help='path for model weights')
    parser.add_argument('-w_HQ', '--weight_HQ', type=str,
                        default='/data1/wangcong/ICCV23/FeMaSR_HRP_model_g.pth',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results/Fukui_East', help='Output folder')

    parser.add_argument('-m', '--model_type', type=str, default='DPT_Next_ViT_L_384', help='depth estimation type')

    parser.add_argument('-p', '--prompt_number', type=int, default=4, help='Number of prompt')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    weight_path_HQ = args.weight_HQ
    weight_dehaze = args.weight_dehaze
    prompt_number = args.prompt_number


    model_type = args.model_type
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    midas.to(device)
    midas.eval()

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    print('paths', args.input)
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')
        print('img_name',img_name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        b,c,h,w = img_tensor.size()
        # print('img_tensor.size()',img_tensor.size())
        # img_tensor = img_tensor[:,:,:,w//2:]
        # b, c, h, w = img_tensor.size()
        # print('img_tensor.size()----', img_tensor.size())
        for prompt in range(prompt_number):
            os.makedirs(os.path.join(args.output, f'{prompt}', 'wo_prompt'), exist_ok=True)
            os.makedirs(os.path.join(args.output, f'{prompt}', 'depth1'), exist_ok=True)
            os.makedirs(os.path.join(args.output, f'{prompt}', 'depth2'), exist_ok=True)
            os.makedirs(os.path.join(args.output, f'{prompt}', 'w_prompt'), exist_ok=True)
            # set up the model
            HQ_model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=False, scale_factor=args.out_scale).to(
                device)
            HQ_model.load_state_dict(torch.load(weight_path_HQ)['params'], strict=False)
            HQ_model.eval()
            Dehaze_model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=args.out_scale).to(
                device)
            Dehaze_model.load_state_dict(torch.load(weight_dehaze)['params'], strict=False)
            Dehaze_model.eval()
            if prompt == 0:
                print('prompt---%d'%(prompt))
                # output_wo_prompt = Dehaze_model.test(input=img_tensor)
                # _, _, _, _, _, depth_quant_1, _, depth1, _ = HQ_model(input=img_tensor, second_img=img_tensor)
                # _, _, _, _, _, depth_quant_2, _, depth2, _ = HQ_model(input=img_tensor, second_img=output_wo_prompt)

                transform = transform_module(model_type)
                input_transform = transform(img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)
                depth1 = midas(input_transform.to(device))
                with torch.no_grad():
                    output_wo_prompt_img = Dehaze_model.test(input=img_tensor)
                output_wo_prompt = transform(output_wo_prompt_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)
                depth2 = midas(output_wo_prompt.to(device))

                depth1 = (depth1 * 255 / torch.max(depth1))
                depth2 = (depth2 * 255 / torch.max(depth2))

                depth1 = torch.nn.functional.interpolate(
                    depth1.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ) / 255.0

                depth2 = torch.nn.functional.interpolate(
                    depth2.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ) / 255.0
                with torch.no_grad():
                    _, _, _, _, _, depth_quant_1, depth_quant_2, _, _ = HQ_model(input=img_tensor, second_img=img_tensor,depth1=depth1, depth2=depth2)
                    output = Dehaze_model.test(input=img_tensor, depth_quant_1=depth_quant_1, depth_quant_2=depth_quant_2)

                output_img = tensor2img(output)
                output_wo_prompt_img = tensor2img(output_wo_prompt_img)
                depth1_img = tensor2img(depth1)
                depth2_img = tensor2img(depth2)

                save_path_output_wo_prompt = os.path.join(args.output, f'{prompt}', 'wo_prompt', f'{img_name}')
                save_path_depth1 = os.path.join(args.output, f'{prompt}', 'depth1', f'{img_name}')
                save_path_depth2 = os.path.join(args.output, f'{prompt}', 'depth2', f'{img_name}')
                save_path = os.path.join(args.output, f'{prompt}', 'w_prompt', f'{img_name}')

                imwrite(output_img, save_path)
                imwrite(output_wo_prompt_img, save_path_output_wo_prompt)
                imwrite(depth1_img, save_path_depth1)
                imwrite(depth2_img, save_path_depth2)
                img_tensor = output
            else:
                print('prompt---%d' % (prompt))
                transform = transform_module(model_type)
                input_transform = transform(img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)
                depth1 = midas(input_transform.to(device))
                with torch.no_grad():
                    output_wo_prompt_img = Dehaze_model.test(input=img_tensor)
                output_wo_prompt = transform(output_wo_prompt_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)
                depth2 = midas(output_wo_prompt.to(device))

                depth1 = (depth1 * 255 / torch.max(depth1))
                depth2 = (depth2 * 255 / torch.max(depth2))

                depth1 = torch.nn.functional.interpolate(
                    depth1.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ) / 255.0

                depth2 = torch.nn.functional.interpolate(
                    depth2.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ) / 255.0
                with torch.no_grad():
                    _, _, _, _, _, depth_quant_1, depth_quant_2, _, _ = HQ_model(input=img_tensor, second_img=img_tensor,
                                                                                 depth1=depth1, depth2=depth2)
                    output = Dehaze_model.test(input=img_tensor, depth_quant_1=depth_quant_1, depth_quant_2=depth_quant_2)

                output_img = tensor2img(output)
                output_wo_prompt_img = tensor2img(output_wo_prompt_img)
                depth1_img = tensor2img(depth1)
                depth2_img = tensor2img(depth2)

                save_path_output_wo_prompt = os.path.join(args.output, f'{prompt}', 'wo_prompt', f'{img_name}')
                save_path_depth1 = os.path.join(args.output, f'{prompt}', 'depth1', f'{img_name}')
                save_path_depth2 = os.path.join(args.output, f'{prompt}', 'depth2', f'{img_name}')
                save_path = os.path.join(args.output, f'{prompt}', 'w_prompt', f'{img_name}')

                imwrite(output_img, save_path)
                imwrite(output_wo_prompt_img, save_path_output_wo_prompt)
                imwrite(depth1_img, save_path_depth1)
                imwrite(depth2_img, save_path_depth2)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
