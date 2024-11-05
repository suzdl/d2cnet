import numpy as np
from matplotlib.colors import Normalize
import torch.nn as nn
import torch
from torchvision import transforms
import math
import random
import torchvision
from models_ori.CBAM import CBAM
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2),
        16: (20, 16, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, in_feat, kernel_size, reduction, bias=True, bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(in_feat, out_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        if in_feat != out_feat:
            self.conv_last = conv(in_feat, out_feat, 1)
        else:
            self.conv_last = None

    def forward(self, x):
        res = self.body(x)
        if self.conv_last is not None:
            x = self.conv_last(x)
        res += x
        return res


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


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


class SUFT(nn.Module):
    def __init__(self, dp_feats, add_feats, scale):
        super(SUFT, self).__init__()
        self.fliper = transforms.RandomHorizontalFlip(1)
        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.dpf_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.total_down = DenseProjection(dp_feats + add_feats, dp_feats + add_feats, scale, up=False, bottleneck=False)
        self.conv_du = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

    def forward(self, depth, rgb):
        dpf = self.fliper(depth)
        dp_h = self.dp_up(depth)
        dpf_h = self.dpf_up(dpf)
        dif = torch.abs(dp_h - self.fliper(dpf_h))

        dif_avg = torch.mean(dif, dim=1, keepdim=True)
        dif_max, _ = torch.max(dif, dim=1, keepdim=True)
        attention = self.conv_du(torch.cat([dif_avg, dif_max], dim=1))
        max = torch.max(torch.max(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)
        min = torch.min(torch.min(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)

        attention = (attention - min) / (max - min + 1e-12)  # [b, 1, h, w]
        rgb_h = rgb * attention
        total = torch.cat([dp_h, rgb_h], dim=1)
        out = self.total_down(total)
        return out

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out


class ConvGroup(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_bn: bool):
        super().__init__()

        # (Conv2d, BN, GELU)
        dim = conv.out_channels
        self.group = nn.Sequential(
            conv,
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.GELU(),
        )

    def forward(self, x):
        return self.group(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DGroup(nn.Module):
    """
    [channels: dim, s] -> DGroup -> [channels: 1, s]
    """

    def __init__(self, in_c: int, out_c: int, dim: int, k_size: int, use_bn: bool):
        super().__init__()

        # conv_d: [dim] -> [1]
        self.conv_d = nn.ModuleList([
            ConvGroup(nn.Conv2d(in_c, dim, kernel_size=k_size, padding='same', dilation=(i + 1)), use_bn=use_bn)
            for i in range(3)
        ])
        # conv_s: [3] -> [1]
        self.conv_s = nn.Sequential(
            nn.Conv2d(3 * dim, out_c, kernel_size=3, padding='same'),
            nn.PReLU(),
        )

        self.attention = eca_layer(3 * dim)

    def forward(self, x):
        f_in = x
        # conv_d
        f_x = [conv(f_in) for conv in self.conv_d]
        # suffix
        f_c = torch.cat(f_x, dim=1)
        f_t = self.attention(f_c)
        f_out = self.conv_s(f_t)
        return f_out


class InceptionDW(nn.Module):
    def __init__(self, in_channels):
        super(InceptionDW, self).__init__()

        cnt = int(in_channels * 0.25)
        self.dwconv3 = nn.Conv2d(cnt, cnt, 3, padding=1, groups=cnt)
        self.dwconv5 = nn.Conv2d(cnt, cnt, 5, padding=2, groups=cnt)
        self.dwconv1 = nn.Conv2d(in_channels - 2 * cnt, in_channels - 2 * cnt, 1, padding=0,
                                 groups=in_channels - 2 * cnt)
        self.split_index = (in_channels - 2 * cnt, cnt, cnt)

        self.attention = eca_layer(in_channels)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        x1, x3, x5 = torch.split(x, self.split_index, dim=1)

        out = torch.cat((self.act(self.dwconv1(x1)), self.act(self.dwconv3(x3)), self.act(self.dwconv5(x5))), dim=1)
        return self.attention(out) + x


# ReCoNet Fuse
class Fuser(nn.Module):
    def __init__(self, in_channels, scale, use_bn):
        super(Fuser, self).__init__()

        # attention layer: [2] -> [1], [2] -> [1]
        self.att_a_conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)
        self.att_b_conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)

        # dilation fuse
        self.decoder = DGroup(in_c=3, out_c=1, dim=in_channels, k_size=3, use_bn=use_bn)
        self.dp_up = DenseProjection(in_channels, in_channels, scale, up=True, bottleneck=False)

    def forward(self, i_in, init_f):
        # recurrent subnetwork
        # generate f_0 with initial function
        depth, image = i_in
        depth_up = self.dp_up(depth)
        i_f = [(depth_up + image) / 2] if init_f is None else init_f

        # loop in subnetwork
        i_f_x, att_a_x, att_b_x = self._sub_forward(depth_up, image, i_f[-1])
        # return as expected
        return i_f_x

    def _sub_forward(self, image, depth, fea_before):
        # attention
        att_a = self._attention(self.att_a_conv, image, fea_before)
        att_b = self._attention(self.att_b_conv, depth, fea_before)

        # focus on attention
        i_1_w = image * att_a
        i_2_w = depth * att_b

        # dilation fuse
        i_in = torch.cat([i_1_w, fea_before, i_2_w], dim=1)
        i_out = self.decoder(i_in)

        # return fusion result of current recurrence
        return i_out, att_a, att_b

    @staticmethod
    def _attention(att_conv, i_a, i_b):
        i_in = torch.cat([i_a, i_b], dim=1)
        i_max, _ = torch.max(i_in, dim=1, keepdim=True)
        i_avg = torch.mean(i_in, dim=1, keepdim=True)
        i_in = torch.cat([i_max, i_avg], dim=1)
        i_out = att_conv(i_in)
        return torch.sigmoid(i_out)


# RASG中的SA模块
class SA(nn.Module):
    def __init__(self, dp_feats, add_feats, scale):
        super(SA, self).__init__()
        self.rgb_conv1 = ConvBlock(add_feats, add_feats, 1, 1, 0, activation='prelu')
        self.rgb_conv1x1 = ConvBlock(add_feats, add_feats, 1, 1, 0, activation='prelu')
        self.rgb_conv3x3 = ConvBlock(add_feats, add_feats, 3, 1, 1, activation='prelu')

        self.dp_conv1x1 = ConvBlock(dp_feats, dp_feats, 1, 1, 0, activation='prelu')
        self.dp_conv3x3 = ConvBlock(dp_feats, dp_feats, 3, 1, 1, activation='prelu')

        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.conv_before = ResidualGroup(default_conv, dp_feats + add_feats, dp_feats + add_feats, 3, reduction=16,
                                         n_resblocks=4)
        self.cbam = CBAM(dp_feats)
        # 定义可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.total_down = DenseProjection(dp_feats + add_feats, dp_feats + add_feats, scale, up=False, bottleneck=False)

    def forward(self, depth, rgb):
        dp_up = self.dp_up(depth)
        rgb_diff = torch.abs(self.rgb_conv3x3(rgb) - self.rgb_conv1x1(rgb))
        dp_diff = torch.abs(self.dp_conv3x3(dp_up) - self.dp_conv1x1(dp_up))
        total_diff = rgb_diff + dp_diff
        attention = self.cbam(total_diff)
        rgb_feature = attention + self.rgb_conv1(rgb) * self.alpha
        out = self.total_down(torch.cat([rgb_feature, dp_up], dim=1))

        return out


#
class SAMulti(nn.Module):
    def __init__(self, dp_feats, add_feats, before_feats, scale):
        super(SAMulti, self).__init__()
        self.rgb_conv1 = ConvBlock(add_feats, add_feats, 1, 1, 0, activation='prelu')
        self.rgb_conv3x3 = ConvBlock(add_feats, add_feats, 3, 1, 1, activation='prelu')
        self.rgb_conv5x5 = ConvBlock(add_feats, add_feats, 5, 1, 2, activation='prelu')

        self.dp_conv1 = ConvBlock(add_feats, add_feats, 1, 1, 0, activation='prelu')
        self.dp_conv3x3 = ConvBlock(dp_feats, dp_feats, 3, 1, 1, activation='prelu')
        self.dp_conv5x5 = ConvBlock(dp_feats, dp_feats, 5, 1, 2, activation='prelu')
        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)

        self.before_conv3x3 = ConvBlock(dp_feats, dp_feats, 3, 1, 1, activation='prelu')
        self.before_conv5x5 = ConvBlock(dp_feats, dp_feats, 5, 1, 2, activation='prelu')
        self.before_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)

        self.conv_last = ResidualGroup(default_conv, before_feats * 2, before_feats * 2, 3, reduction=16, n_resblocks=4)
        self.cbam = CBAM(dp_feats)
        # 定义可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))

        self.total_down = DenseProjection(before_feats * 2, before_feats * 2, scale, up=False, bottleneck=False)
        self.decoder = InceptionDW(in_channels=before_feats * 2)

    def forward(self, depth, rgb, before):
        if before is None:
            before = depth
        dp_up = self.dp_up(depth)
        before_up = self.before_up(before)
        rgb_diff = torch.abs(self.rgb_conv3x3(rgb) - self.rgb_conv5x5(rgb))
        dp_diff = torch.abs(self.dp_conv3x3(dp_up) - self.dp_conv5x5(dp_up))
        before_diff = torch.abs(self.before_conv3x3(before_up) - self.before_conv5x5(before_up))

        total_diff = rgb_diff + dp_diff + before_diff
        attention = self.cbam(total_diff)
        rgb_feature = attention + (self.rgb_conv1(rgb) * self.alpha) + (self.dp_conv1(dp_up) * self.beta)
        out = self.total_down(torch.cat([rgb_feature, before_up], dim=1))

        return self.decoder(out)

# RASG中的SA模块 同时使用parallel conv融合前一阶段的特征
class SAFuseBefore(nn.Module):
    def __init__(self, dp_feats, rgb_feats, before_feats, scale):
        super(SAFuseBefore, self).__init__()
        self.rgb_conv1 = ConvBlock(rgb_feats, rgb_feats, 1, 1, 0, activation='prelu')
        self.rgb_conv1x1 = ConvBlock(rgb_feats, rgb_feats, 1, 1, 0, activation='prelu')
        self.rgb_conv3x3 = ConvBlock(rgb_feats, rgb_feats, 3, 1, 1, activation='prelu')

        self.dp_conv1x1 = ConvBlock(dp_feats, dp_feats, 1, 1, 0, activation='prelu')
        self.dp_conv3x3 = ConvBlock(dp_feats, dp_feats, 3, 1, 1, activation='prelu')

        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.cbam = CBAM(dp_feats)
        # 定义可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.total_down = DenseProjection(dp_feats + rgb_feats, dp_feats + rgb_feats, scale, up=False, bottleneck=False)
        if before_feats > 0:
            self.decoder = DGroup(in_c=dp_feats + rgb_feats + before_feats, out_c=dp_feats + rgb_feats,
                                  dim=(dp_feats + rgb_feats + before_feats) // 2, k_size=3, use_bn=False)
            self.conv_before = ResidualGroup(default_conv, before_feats, before_feats, 3, reduction=16, n_resblocks=4)
        else:
            self.conv_before = None
            self.decoder = None

    def forward(self, depth, rgb, before=None):

        dp_up = self.dp_up(depth)
        rgb_diff = torch.abs(self.rgb_conv3x3(rgb) - self.rgb_conv1x1(rgb))
        dp_diff = torch.abs(self.dp_conv3x3(dp_up) - self.dp_conv1x1(dp_up))
        total_diff = rgb_diff + dp_diff
        attention = self.cbam(total_diff)
        rgb_feature = attention + self.rgb_conv1(rgb) * self.alpha
        out = self.total_down(torch.cat([rgb_feature, dp_up], dim=1))
        if before is not None:
            before_feat = self.conv_before(before)
            out = self.decoder(torch.cat([out, before_feat], dim=1))
        return out


# CODON中的MC unit
class MCUnit(torch.nn.Module):
    def __init__(self, num_features):
        super(MCUnit, self).__init__()
        self.conv_5x5 = ConvBlock(num_features, num_features, 5, padding=2, activation='prelu')
        self.conv_3x3 = ConvBlock(num_features, num_features, 3, padding=1, activation='prelu')

        self.cat = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            ConvBlock(num_features * 2, num_features * 2, 5, padding=2, activation=None),
            ConvBlock(num_features * 2, num_features, 1, padding=0, activation=None)
        ])

    def forward(self, inputs):
        out = torch.cat((self.conv_3x3(inputs), self.conv_5x5(inputs)), dim=1)
        return self.cat(out)


# RASG中的SA模块，和前一阶段的特种cat之后用使用CAC中的MC unit
class SABeforeMC(nn.Module):
    def __init__(self, dp_feats, rgb_feats, before_feats, scale):
        super(SABeforeMC, self).__init__()
        self.rgb_conv1 = ConvBlock(rgb_feats, rgb_feats, 1, 1, 0, activation='lrelu')
        self.rgb_conv1x1 = ConvBlock(rgb_feats, rgb_feats, 1, 1, 0, activation='lrelu')
        self.rgb_conv3x3 = ConvBlock(rgb_feats, rgb_feats, 3, 1, 1, activation='lrelu')

        self.dp_conv1x1 = ConvBlock(dp_feats, dp_feats, 1, 1, 0, activation='lrelu')
        self.dp_conv3x3 = ConvBlock(dp_feats, dp_feats, 3, 1, 1, activation='lrelu')

        self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.cbam = CBAM(dp_feats)
        # 定义可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.total_down = DenseProjection(dp_feats + rgb_feats, dp_feats + rgb_feats, scale, up=False, bottleneck=False)
        if before_feats > 0:
            self.conv_before = ResidualGroup(default_conv, before_feats, before_feats, 3, reduction=16, n_resblocks=4)
            self.mc = MCUnit(before_feats + dp_feats + rgb_feats)
        else:
            self.conv_before = None
            self.decoder = None

    def forward(self, depth, rgb, before=None):
        dp_up = self.dp_up(depth)
        rgb_diff = torch.abs(self.rgb_conv3x3(rgb) - self.rgb_conv1x1(rgb))
        dp_diff = torch.abs(self.dp_conv3x3(dp_up) - self.dp_conv1x1(dp_up))
        total_diff = rgb_diff + dp_diff
        attention = self.cbam(total_diff)
        rgb_feature = attention + self.rgb_conv1(rgb) * self.alpha
        out = self.total_down(torch.cat([rgb_feature, dp_up], dim=1))

        # parallel conv
        if before is not None:
            before_feat = self.conv_before(before)
            out = self.mc(torch.cat([out, before_feat], dim=1))
        return out


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                 2).transpose(
                0, 1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


# LIC_TCM
class ConvTransBlock(nn.Module):
    def __init__(self, dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(dim, dim, self.head_dim, self.window_size, self.drop_path, self.type)
        # self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)

        self.conv_block = ResBlock(default_conv, dim, 3, act=nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        # conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(x) + x
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

if __name__ == "__main__":
    x = torch.randn((1, 32, 480, 640)).cuda()
    y = torch.randn((1, 32, 60, 80)).cuda()
    # z = torch.randn((1, 32, 60, 80)).cuda()
    z = None
    fuse = SAMulti(dp_feats=32, add_feats=32, before_feats=32, scale=8).cuda()
    out = fuse(y, x, z)
    print(out.shape)

    # featrue = torch.randn((1, 32, 480, 640)).cuda()
    # block = ConvTransBlock(16, 16, 8, 8, 0).cuda()
    # out = block(featrue)
    # print(out.shape)

