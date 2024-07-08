import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math

from basicsr.utils.registry import ARCH_REGISTRY

import torchvision
from .network_swinir import RSTB
from .fema_utils import ResBlock, CombineQuantBlock
from timm.models.layers import DropPath
from .vgg_arch import VGGFeatureExtractor
from distutils.version import LooseVersion
from .opsdcn.dcn import ModulatedDeformConvPack, modulated_deform_conv
from .utilsdcn import get_root_logger


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # print('fsdfsf')

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, self.groups, self.deformable_groups)


class DCNv2Pack_fusion(nn.Module):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.

    """
    def __init__(self, in_channel1, in_channel2, out_channel, kernel=3, padding=1):
        super(DCNv2Pack_fusion, self).__init__()
        self.fusion1 = nn.Conv2d(in_channel1 + in_channel2, out_channel, 1, 1)
        self.fusion2 = nn.Conv2d(in_channel1 + in_channel2, out_channel, 1, 1)

        self.DCNv2Pack1 = DCNv2Pack(out_channel, out_channel, kernel, padding=1)
        self.DCNv2Pack2 = DCNv2Pack(out_channel, out_channel, kernel, padding=1)

        self.fusion = nn.Conv2d(2 * out_channel, out_channel, 1, 1)

    def forward(self, x, feat):
        fusion1 = self.fusion1(torch.cat([x, feat], dim=1))
        DCNv2Pack1 = self.DCNv2Pack1(x, fusion1)

        fusion2 = self.fusion2(torch.cat([feat, x], dim=1))
        DCNv2Pack2 = self.DCNv2Pack2(fusion2,x)
        fusion = self.fusion(torch.cat([DCNv2Pack1, DCNv2Pack2], dim=1))

        return fusion


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
               torch.sum(y ** 2, dim=1) - 2 * \
               torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean()
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                 blk_depth=6,
                 num_heads=8,
                 window_size=8,
                 **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 depth_guide='input_cat',
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3
        if not LQ_stage:
            self.blocks = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            self.max_depth = max_depth
            self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)
            res = input_res
            for i in range(max_depth):
                in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
                tmp_down_block = [
                    nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                ]
                self.blocks.append(nn.Sequential(*tmp_down_block))
                res = res // 2
        else:
            # for i in range(max_depth):
            self.depth_guide = depth_guide

            if self.depth_guide == 'input_cat':
                self.in_conv = nn.Sequential(
                    nn.Conv2d(in_channel * 2, channel_query_dict[input_res], 3, padding=1),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                )
            else:
                self.in_conv = nn.Sequential(
                    nn.Conv2d(in_channel, channel_query_dict[input_res], 3, padding=1),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                )

            self.blocks = nn.ModuleList()
            self.up_blocks = nn.ModuleList()
            self.max_depth = max_depth
            res = input_res
            for i in range(max_depth):
                in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
                tmp_down_block = [
                    nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                ]
                self.blocks.append(nn.Sequential(*tmp_down_block))
                res = res // 2

        self.LQ_stage = LQ_stage

    def forward(self, input, depth=None):
        if not self.LQ_stage:
            b, c, h, w = input.size()
            outputs = []
            x = self.in_conv(input)
            x = F.upsample(x, (h, w))
            for idx, m in enumerate(self.blocks):
                x = m(x)
                outputs.append(x)
            return outputs
        else:
            outputs = []
            if depth is not None:
                x = self.in_conv(torch.cat([input, depth], dim=1))
            else:
                # print('input',input.size())
                x = self.in_conv(input)
            for idx, m in enumerate(self.blocks):
                x = m(x)
                outputs.append(x)
            return outputs


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, prompt=None, H=None, W=None):
        B, N, C = x.shape
        if prompt is not None:
            q = x + prompt
        else:
            q = x
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, prompt=None, H=None, W=None):

        if prompt is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), prompt, H, W))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


@ARCH_REGISTRY.register()
class FeMaSRNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=1,
                 use_residual=True,
                 position='depth',
                 prompt=True,
                 n_layers=16,
                 depth_guide='prompt',
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)
        self.prompt = prompt

        self.position = position

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual

        self.depth_guide = depth_guide

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        self.dcn_fusion = nn.ModuleList()
        self.dcn_fusion.append(DCNv2Pack_fusion(in_channel1=channel_query_dict[32], in_channel2=channel_query_dict[32],
                                                out_channel=channel_query_dict[32]))
        # self.dcn_fusion.append()
        self.dcn_fusion.append(DCNv2Pack_fusion(in_channel1=channel_query_dict[32], in_channel2=channel_query_dict[64],
                                             out_channel=channel_query_dict[64]))
        self.dcn_fusion.append(DCNv2Pack_fusion(in_channel1=channel_query_dict[32], in_channel2=channel_query_dict[128],
                                             out_channel=channel_query_dict[128]))

        if LQ_stage:
            res = 32
            self.upsampler = nn.ModuleList()
            in_channel_texture, out_channel_texture = channel_query_dict[res], channel_query_dict[res]
            self.upsampler.append(nn.Sequential(
                    ResBlock(out_channel_texture, out_channel_texture, norm_type, act_type),
                    ResBlock(out_channel_texture, out_channel_texture, norm_type, act_type)))
            res = 64
            for i in range(2):
                out_channel_texture, out_channel_texture = channel_query_dict[res], channel_query_dict[res]
                self.upsampler.append(nn.Sequential(
                    ResBlock(out_channel_texture, out_channel_texture, norm_type, act_type),
                    ResBlock(out_channel_texture, out_channel_texture, norm_type, act_type),
                    )
                )
                res = res * 2

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))  ## 3 ##
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))

        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            encode_depth,
            self.gt_res // self.scale_factor,
            channel_query_dict,
            norm_type, act_type, LQ_stage, depth_guide=self.depth_guide
        )

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # self.depth_guide = depth_guide
        self.dim_embd = channel_query_dict[32]
        self.n_head = 8
        self.n_layers = n_layers
        #####################################################
        self.transformer = nn.Sequential(
            *[Block(dim=self.dim_embd, num_heads=self.n_head)
              for _ in range(self.n_layers)])

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

    def _get_pos_embed(self, H, W):
        return F.interpolate(
            self.pos_embed.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def encode_and_decode(self, input, gt_img=None, depth1=None, depth2=None, depth_quant_1=None, depth_quant_2=None,
                          gt_indices=None, current_iter=None):
        codebook_loss_list = []
        indices = None

        h, c, h, w = input.size()

        codebook_loss_list_second = []
        indices_second = None

        if self.LQ_stage is True and self.depth_guide == 'input_cat':
            depth1 = torch.cat([depth1, depth1, depth1], dim=1)  # /255.0
            depth1 = torch.clip(depth1, 0, 1)
            enc_feats1 = self.multiscale_encoder(input.detach(), depth=depth1)
        else:
            enc_feats1 = self.multiscale_encoder(input.detach())
        enc_feats1 = enc_feats1[::-1]

        x1 = enc_feats1[0]

        z_quant = x1
        # codebook_loss_list.append(codebook_loss)

        ###################################################
        if not self.LQ_stage:
            input1 = gt_img
            input2 = input
            if input1 is not None:
                depth1 = torch.cat([depth1, depth1, depth1], dim=1)  # /255.0
                depth1 = torch.clip(depth1, 0, 1)
                enc_feats1_depth1 = self.multiscale_encoder(depth1)
                enc_feats1_depth1 = enc_feats1_depth1[::-1]

                x1_depth1 = enc_feats1_depth1[0]
                # x1_depth1 = self.before_quant_group[0](x1_depth1)
                depth_quant_1 = x1_depth1

            if input2 is not None:
                depth2 = torch.cat([depth2, depth2, depth2], dim=1)  # /255.0
                depth2 = torch.clip(depth2, 0, 1)
                enc_feats1_depth2 = self.multiscale_encoder(depth2)
                enc_feats1_depth2 = enc_feats1_depth2[::-1]

                x1_depth2 = enc_feats1_depth2[0]
                # x1_depth2 = self.before_quant_group[0](x1_depth2)
                depth_quant_2 = x1_depth2

        ########################################
        device = z_quant.device
        ########################################

        depth_error = torch.abs(depth_quant_1 - depth_quant_2)
        prompt = depth_error * z_quant

        x_ori = z_quant
        x = z_quant
        b, c, h, w = x.size()
        # print('x',x.size())
        x = x.flatten(2).transpose(1, 2)
        prompt = prompt.flatten(2).transpose(1, 2)

        ###########  prompt embedding  ###################

        x = x + prompt
        ###################### transformer #################
        for blk in self.transformer:
            x = blk(x, prompt, h, w)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        z_quant = x
        trans_feature = z_quant

        z_quant = self.before_quant_group[0](z_quant)
        ###################################################################

        # if self.LQ_stage is True:
        z_quant_2, codebook_loss_2, indices_2 = self.quantize_group[0](z_quant, gt_indices)
        indices_second = indices_2
        codebook_loss_list_second.append(codebook_loss_2)
        z_quant = z_quant_2
        ####################################################

        quant = z_quant
        x = self.after_quant_group[0](quant)


        b,c,h,w = x.size()
        if not self.LQ_stage:
            for i in range(self.max_depth):
                x = self.decoder_group[i](x)
        else:
            for i in range(self.max_depth):
                if self.use_residual:
                    upsampler = F.upsample(trans_feature, (h * (2 ** i), w * (2 ** i)))
                    dcn_fusion = self.dcn_fusion[i](enc_feats1[i], upsampler)
                    upsampler = self.upsampler[i](dcn_fusion)
                    x = x + upsampler
                else:
                    x = x
                x = self.decoder_group[i](x)

        out_img = self.out_conv(x)

        codebook_loss = sum(codebook_loss_list) if not len(codebook_loss_list) else 0
        codebook_loss_second = sum(codebook_loss_list_second)

        return out_img, codebook_loss, codebook_loss_second, indices, indices_second, depth_quant_1, depth_quant_2, depth1, depth2  # , depth1_reconstruction, depth2_reconstruction

    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, padding=8):
        _, _, h, w = x.size()
        # print('_, _, h, w = x.size()',x.size())
        mod_pad_h = (padding - h % (padding)) % (padding)
        mod_pad_w = (padding - w % (padding)) % (padding)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
        return x

    @torch.no_grad()
    def test(self, input, second_img=None, depth1=None, depth2=None, depth_quant_1=None, depth_quant_2=None):

        _, _, h_old, w_old = input.size()

        input = self.check_image_size(input, padding=8)

        dec, _, _, _, _, _, _, depth1, depth2 = self.encode_and_decode(input=input, gt_img=second_img, depth1=depth1,
                                                                       depth2=depth2, depth_quant_1=depth_quant_1,
                                                                       depth_quant_2=depth_quant_2)

        output = dec
        output = output[..., :h_old, :w_old]

        return output

    def forward(self, input, second_img=None, depth1=None, depth2=None, depth_quant_1=None, depth_quant_2=None,
                gt_indices=None):

        if gt_indices is not None:
            dec, codebook_loss, codebook_loss_second, indices, indices_second, depth_quant_1, depth_quant_2, depth1, depth2 = self.encode_and_decode(
                input, gt_img=second_img, depth1=depth1, depth2=depth2, depth_quant_1=depth_quant_1,
                depth_quant_2=depth_quant_2, gt_indices=gt_indices)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, codebook_loss, codebook_loss_second, indices, indices_second, depth_quant_1, depth_quant_2, depth1, depth2 = self.encode_and_decode(
                input, gt_img=second_img, depth1=depth1, depth2=depth2, depth_quant_1=depth_quant_1,
                depth_quant_2=depth_quant_2)

        return dec, codebook_loss, codebook_loss_second, indices, indices_second, depth_quant_1, depth_quant_2, depth1, depth2
