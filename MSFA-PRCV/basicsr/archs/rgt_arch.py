import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import functional as F, Parameter, Softmax,  Module
from einops import rearrange, reduce
from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import math
import numpy as np

import random

from basicsr.utils.registry import ARCH_REGISTRY


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x

class ChannelAttentionMoudle(nn.Module):
    """ Channel Attention Moudle
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)
        self.TheFirstBranch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.TheSecondBranch = nn.Sequential(
            nn.Conv2d(dim // 6, dim // 6, kernel_size=3, stride=1, padding=1, groups=dim // 6),
            nn.Conv2d(dim // 6, dim // 6, 7, stride=1, padding=9, groups=dim // 6, dilation=3),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.pro_out = nn.Conv2d(dim // 6, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        res = x
        TheFirstBranch = self.TheFirstBranch(x)
        TheSecondBranch = self.TheSecondBranch(x)
        out = self.pro_out(res * TheFirstBranch * TheSecondBranch)
        return out



class SpatialAttentionBlock(nn.Module):
    """ Spatial Attention Block.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 5, 1, 1, 0)
        self.dwconv = nn.Conv2d(dim // 5,  dim // 5, kernel_size=3, stride=1, padding=1, groups= dim // 5)
        self.pro_out = nn.Conv2d(dim // 10, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.pro_out(x)
        return x
class FrequencyAttentionBlock(nn.Module):
    """ Frequency Attention Block.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.act = nn.GELU()
        self.res_2 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.GELU()
        )
        self.conv_out = nn.Conv2d(dim // 2, dim, 1, 1, 0)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        res = x
        x = self.conv_1(x)
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.act(x1), self.res_2(x2)), dim=1)
        out = self.conv_out(out)
        return out + res
class HAB(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """ Channel Transposed Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.channel_AttentionMoudle = ChannelAttentionMoudle(dim)
        self.spatial_AttentionBlock = SpatialAttentionBlock(dim)
        self.frequence_AttentionBlock = FrequencyAttentionBlock(dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape

        # # dwconv and channel attention block
        conv_x = self.dwconv(x.reshape(B,H,W,C).permute(0,3,1,2))
        attened_x=x
        # Channel Attention Module
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        channel_map = self.channel_AttentionMoudle(attention_reshape)
        attened_x = conv_x.permute(0,2,3,1).reshape(B,N,C) + channel_map.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean')

        # spatial_AttentionBlock and frequence_AttentionBlock
        spatial_map = self.spatial_AttentionBlock(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, C)
        frequenece_map = self.frequence_AttentionBlock(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, C)

        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        #feature fusion
        x = attened_x + conv_x + frequenece_map

        x = self.proj(x)

        x = self.proj_drop(x)

        return x
class L_SA(nn.Module):
    # The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py
    def __init__(self, dim, num_heads,
                 split_size=[2,4], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., idx=0, reso=64, rs_id=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                WindowAttention(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """

        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)


        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            # H-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# class GCA(nn.Module):
#     """
#     Tips:
#         Mainly borrows from SKNet (https://github.com/implus/SKNet)
#     """
#     def __init__(self, dim):
#         super().__init__()
#         # K = d*(k_size-1)+1
#         # (H - k_size + 2padding)/stride + 1
#         # (5,1)-->(7,3)
#         # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # K=5, 64-5+4+1=64
#         # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#
#         # (3,1)-->(5,2)
#         self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  #
#         self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) # K=9, 64-9+8 + 1
#
#         # (5,1)-->(7,4)
#         # self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  #
#         # self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=12, groups=dim, dilation=4) # K=25, 64-25+2*12 + 1
#
#         self.conv1 = nn.Conv2d(dim, dim // 2, 1)
#         self.conv2 = nn.Conv2d(dim, dim // 2, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv = nn.Conv2d(dim // 2, dim, 1)
#
#     def forward(self, x):
#         attn1 = self.conv0(x)
#         attn2 = self.conv_spatial(attn1)
#
#         attn1 = self.conv1(attn1)
#         attn2 = self.conv2(attn2)
#
#         attn = torch.cat([attn1, attn2], dim=1)
#         avg_attn = torch.mean(attn, dim=1, keepdim=True)
#         max_attn, _ = torch.max(attn, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         sig = self.conv_squeeze(agg).sigmoid()
#         attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
#         attn = self.conv(attn)
#         return x * attn
# class MSF(nn.Module):
#     def __init__(self, dim):
#         super(MSF, self).__init__()
#
#         # hidden_features = int(dim * ffn_expansion_factor)
#         hidden_features = 180
#
#         self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1)
#
#         self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
#         self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features)
#         self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features)
#
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()
#         self.relu7 = nn.ReLU()
#
#         self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features )
#         self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features )
#         self.dwconv7x7_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3, groups=hidden_features )
#
#         self.relu3_1 = nn.ReLU()
#         self.relu5_1 = nn.ReLU()
#         self.relu7_1 = nn.ReLU()
#
#         self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1)
#
#     def forward(self, x):
#         x = self.project_in(x)
#
#         x1_3, x2_3, x3_3 = self.relu3(self.dwconv3x3(x)).chunk(3, dim=1)
#         x1_5, x2_5, x3_5 = self.relu5(self.dwconv5x5(x)).chunk(3, dim=1)
#         x1_7, x2_7, x3_7 = self.relu7(self.dwconv7x7(x)).chunk(3, dim=1)
#
#         x1 = torch.cat([x1_3, x1_5, x1_7], dim=1)
#         x2 = torch.cat([x2_3, x2_5, x2_7], dim=1)
#         x3 = torch.cat([x3_3, x3_5, x3_7], dim=1)  #
#
#         x1 = self.relu3_1(self.dwconv3x3_1(x1))
#         x2 = self.relu5_1(self.dwconv5x5_1(x2))
#         x3 = self.relu7_1(self.dwconv7x7_1(x3))
#
#         x = torch.cat([x1, x2, x3], dim=1)
#
#         x = self.project_out(x)
#
#         return x
class MSDWConv(nn.Module):

    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dw_sizes = dw_sizes
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(dw_sizes)):
            if i == 0:
                channels = dim - dim // len(dw_sizes) * (len(dw_sizes) - 1)
            else:
                channels = dim // len(dw_sizes)
            conv = nn.Conv2d(channels, channels, kernel_size=dw_sizes[i], padding=dw_sizes[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class MSConvStar(nn.Module):

    def __init__(self, dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = MSDWConv(dim=hidden_dim, dw_sizes=dw_sizes)
        self.fc2 = nn.Conv2d(hidden_dim // 2, dim, 1)
        self.num_head = len(dw_sizes)
        self.act = nn.GELU()

        assert hidden_dim // self.num_head % 2 == 0

    def forward(self, x):

        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)

        return x
class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.depthwise=nn.Conv2d(2*embed_dim, 2*embed_dim, groups=2*embed_dim, kernel_size=3, stride=1, padding=1)
        self.pointwise=nn.Conv2d(2*embed_dim, 2*embed_dim,1,1)
        self.fusion = nn.Conv2d(4*embed_dim, 2*embed_dim, 1, 1)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)
        ftfed1=self.depthwise(ffted)
        ftfed2=self.pointwise(ffted)
        ffted=torch.cat([ftfed1,ftfed2],dim=1)
        ffted=self.fusion(ffted)
        ffted=self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output
class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv2 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1)


    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)

        return output

    def flops(self, H, W):
        total_flops = 0
        # conv1 FLOPs
        total_flops += self.conv1[0].in_channels * self.conv1[0].out_channels * H * W

        # FourierUnit FLOPs
        total_flops += self.fu.flops(H, W)

        # conv2 FLOPs
        total_flops += self.conv2.in_channels * self.conv2.out_channels * H * W

        # last_conv FLOPs
        if self.last_conv is not None:
            total_flops += self.conv2.out_channels * self.conv2.out_channels * 3 * 3 * H * W

        return total_flops
class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim , 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim , embed_dim, 3, 1,1 ),
        )
        self.convv = nn.Conv2d(embed_dim, embed_dim, 1, 1)
    def __call__(self, x):
        shortcut=x
        x=self.body(x)
        x =self.convv( x + shortcut)
        return x
class SFB(nn.Module):
    def __init__(self, embed_dim,red=1):
        super(SFB, self).__init__()
        self.S = FrequencyAttentionBlock(embed_dim )
        self.F = SpectralTransform(embed_dim )
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1)
    def __call__(self, x):

        s = self.S(x)
        f = self.F(x)
        out = self.fusion(torch.cat([s, f], dim=1))

        return out

    def flops(self, H, W):
        total_flops = 0
        # Calculate FLOPs for ResB
        total_flops += self.S.flops(H, W)
        # Calculate FLOPs for SpectralTransform
        total_flops += self.F.flops(H, W)
        # Calculate FLOPs for the fusion conv layer
        total_flops += self.fusion.in_channels * self.fusion.out_channels * H * W  # 1x1 conv

        return total_flops
class CSB(nn.Module):
    def __init__(self, dim, x_size):
        super(CSB, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 4, stride=4)
        self.proj = nn.Linear(dim*2, dim*2)
        self.LN=nn.LayerNorm(dim * 2)
        self.sm = nn.Softmax(dim=-2)
    def SpatialCorrection(self, x, x_size):
        B, N, C = x.shape
        H, W = x_size
        x1, x2 = x.chunk(2, dim = 2)#(n,c//2)
        x2 = self.conv(x2.permute(0, 2, 1).view(B, C // 2, H, W).contiguous()).view(B, C//2, N//16)
        x3 = torch.sigmoid(x1 @ x2 )#(n,n//16)
        x3 = x3 @ x2.permute(0, 2, 1).contiguous()#(n ,c//2)
        return x3

    def ChannelCorrection(self, x, x_size):
        x1, x2 = x.chunk(2, dim=2)
        x3 = self.sm(x1.permute(0, 2, 1).contiguous() @ x2)#(c//2, c//2)
        x3 = x2 @ x3
        return  x3
    def forward(self, x, x_size):
        x1 = self.SpatialCorrection(x, x_size)
        x2 = self.ChannelCorrection(x, x_size)
        x3 = self.proj(self.LN(torch.cat([x1, x2], dim = 2)))
        return  x3.permute(0, 2, 1).contiguous()



class RG_SA(nn.Module):
    """
    Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5, scales=[1,3,5]):
        super(RG_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scales = scales
        self.cr = int(dim * c_ratio)  # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        # self.reduction1 = nn.Conv2d(dim//2, dim//2, kernel_size=4, stride=4, groups=dim//2)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),
            nn.GELU())

        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reductions = nn.ModuleList([
            nn.Conv2d(dim//3, dim//3, kernel_size=s,  padding= s//2, groups=dim//3)
            for s in scales
        ])
        self.projc = nn.Linear(dim, self.cr)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(dim , self.cr, 4, stride=4),
            nn.GELU(),

        )
    def forward(self, x, H, W):
        B, N, C = x.shape

        _scale = 1

        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        res = _x
        _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))


        _scale = 4 ** _time
        multi_scale_feats = []

        _x1,_x2,_x3 = _x.chunk(3, dim = 1)
        _xlist = [_x1,_x2,_x3]
        # Recursion xT
        for i, scale in enumerate(self.scales):

            # 每个尺度独立递归
            for _ in range(i + 1):  # 递归次数与尺度层级相关
                _xlist[i] = self.reductions[i](_xlist[i])
            multi_scale_feats.append(_xlist[i])

            # 多尺度特征融合
        fused_feats = torch.cat(multi_scale_feats, dim=1)  # [B, C*num_scales, H', W']
        _x = self.fusion(fused_feats).flatten(2).permute(0, 2, 1)  # [B, N', C']
        # _x = _x.view(B,H,W,C//2).permute(0,3, 1, 2).contiguous()
        # for _ in range(_time):
        #     _x = self.reduction1(_x)
        # _x = _x.permute(0, 2,3,1).view(B,N//_scale**2,C//2).contiguous()
        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)

        # corss-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE
        # v_shape=(B, H, N', C//H)
        v = v + self.cpe(
            v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale , W // _scale)).view(
            B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class GCAM(Module):
    """ Channel attention module"""
    def __init__(self, in_dim,reduction_ratio=16):
        super(GCAM, self).__init__()
        self.chanel_in = in_dim
        self.reduction_ratio = reduction_ratio

        # 第一层和第二层卷积，用于通道压缩和扩展
        self.conv1 = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_dim // reduction_ratio, in_dim, kernel_size=1, padding=0)
        # self.conv3=nn.Conv2d(in_dim,in_dim,1,1)
        self.fusion=nn.Conv2d(2*in_dim,in_dim,1,1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        shortcut=x
        #x=x.permute(0,3,1,2).contiguous()
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        # x=x.permute(0,2,3,1).contiguous()
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        #energy_new =energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        out=torch.cat([out,shortcut],dim=1)
        # out = out.permute(0, 3, 1, 2).contiguous()
        out=self.fusion(out)
        # out = out.permute(0, 2, 3, 1).contiguous()
        return out

    def flops(self, H, W):
        # Initial FLOPs from the convolutions
        flops = 0
        flops += self.chanel_in * (self.chanel_in // self.reduction_ratio) * H * W  # conv1
        flops += (self.chanel_in // self.reduction_ratio) * self.chanel_in * H * W  # conv2
        flops += 2 * self.chanel_in * self.chanel_in * H * W  # fusion conv

        # Compute the FLOPs for the attention mechanism
        flops += self.chanel_in * H * W * (self.chanel_in * H * W)  # energy calculation: bmm
        flops += self.chanel_in * H * W * (self.chanel_in * H * W)  # attention application: bmm

        return flops
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
                 rs_id=0, split_size=[2,4], shift_size=[1,2], reso=64, c_ratio=0.5, layerscale_value=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if idx % 2 == 0:
            self.attn = L_SA(
                dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                drop=drop, idx=idx, reso=reso, rs_id=rs_id
            )
        else:
            self.attn = RG_SA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, c_ratio=c_ratio
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        # HAI
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_size):
        H , W = x_size

        res = x

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # HAI
        x = x + (res * self.gamma)

        return x

class ResidualGroup(nn.Module):

    def __init__(   self,
                    dim,
                    reso,
                    num_heads,
                    mlp_ratio=4.,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_paths=None,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    depth=2,
                    use_chk=False,
                    resi_connection='1conv',
                    rs_id=0,
                    split_size=[8,8],
                    c_ratio = 0.5):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso
        self.cr = int(dim * c_ratio)
        self.attnblock = SFB(dim)
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                idx = i,
                rs_id = rs_id,
                split_size = split_size,
                shift_size = [split_size[0]//2, split_size[1]//2],
                c_ratio = c_ratio,
                )for i in range(depth)])


        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.i = 0
        layerscale_value = 1e-4
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        i = self.i
        reslist = []
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
            reslist.append(x)
            if i % 2 != 0:
                reslist[i - 1] =rearrange( reslist[i-1].permute(0 , 2, 1).contiguous(), "b c (h w) -> b c h w", h=H, w=W)
                x = x +rearrange( self.attnblock(reslist[i-1]),"b c h w -> b (h w) c")*self.gamma
            i = i + 1

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x



class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class RGT(nn.Module):

    def __init__(self,
                img_size=64,
                in_chans=3,
                embed_dim=180,
                depth=[2,2,2,2],
                num_heads=[2,2,2,2],
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_chk=False,
                upscale=2,
                img_range=1.,
                resi_connection='1conv',
                split_size=[8,8],
                c_ratio=0.5,
                **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rs_id=i,
                split_size = split_size,
                c_ratio = c_ratio
                )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, Reconstruction ------------------------- #
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x


if __name__ == '__main__':
    upscale = 1
    height = 62
    width = 66
    model = RGT(
        upscale=2,
        in_chans=3,
        img_size=64,
        img_range=1.,
        depth=[6,6,6,6,6,6],
        embed_dim=180,
        num_heads=[6,6,6,6,6,6],
        mlp_ratio=2,
        resi_connection='1conv',
        split_size=[8, 8],
        upsampler='pixelshuffle').cuda()
    # print(model)
    print(height, width)

    x = torch.randn((1, 3, height, width)).cuda()
    x = model(x)
    print(x.shape)