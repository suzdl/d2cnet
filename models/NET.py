from models_ori.common import *
import random

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision import transforms
from torchvision.utils import save_image
from utils import get_parameter_number


class Net(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(Net, self).__init__()
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResidualGroup(default_conv, num_feats,  num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.rgb_rb3 = ResidualGroup(default_conv, num_feats,  num_feats * 2, kernel_size, reduction=16, n_resblocks=4)
        self.rgb_rb4 = ResidualGroup(default_conv, num_feats * 2,  num_feats * 4, kernel_size, reduction=16, n_resblocks=4)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats,  num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, num_feats, num_feats * 2, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, num_feats * 2, num_feats * 4, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, num_feats * 8, num_feats * 8, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = SAMulti(dp_feats=32, add_feats=32, before_feats=32, scale=scale)
        self.bridge2 = SAMulti(dp_feats=64, add_feats=64, before_feats=64, scale=scale)
        self.bridge3 = SAMulti(dp_feats=128, add_feats=128, before_feats=128, scale=scale)

        # self.downsample = default_conv(1, 128, kernel_size=kernel_size)

        my_tail = [
            ResidualGroup(
                default_conv, 256, 256, kernel_size, reduction=16, n_resblocks=4),
            ResidualGroup(
                default_conv, 256, 128,kernel_size, reduction=16, n_resblocks=4)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(256, 256, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(128, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        image, depth = x

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        rgb_in = self.act(self.conv_rgb1(image))
        rgb_1 = self.rgb_rb2(rgb_in)
        fuse1 = self.bridge1(dp1, rgb_1, None) # [64, 60, 80]

        dp2 = self.dp_rg2(dp1)
        rgb2 = self.rgb_rb3(rgb_1)
        fuse2 = self.bridge2(dp2, rgb2, fuse1)

        dp3 = self.dp_rg3(dp2)
        rgb3 = self.rgb_rb4(rgb2)
        fuse3 = self.bridge3(dp3, rgb3, fuse2)

        dp4 = self.dp_rg4(fuse3)

        tail_in = self.upsampler(dp4)

        return self.last_conv(self.tail(tail_in)) + self.bicubic(depth)



if __name__ == '__main__':

    from thop import profile
    # from thop import clever_format

    # with torch.no_grad():
    #     x = torch.randn(1, 3, 256, 256).cuda()
    #     y = torch.randn(1, 1, 16, 16).cuda()
    #     model = Net(num_feats=32, kernel_size=3, scale=16).cuda()
    #     input = (x, y)
    #     macs, params = profile(model, inputs=(input, ))
    #     macs, params = clever_format([macs, params], "%.3f")
    #     print(macs, params)

    #     print(get_parameter_number(model))
    # from ptflops import get_model_complexity_info


    # def prepare_input(resolution):
    #     x = torch.randn(1, 3, 256, 256).cuda()
    #     y = torch.randn(1, 1, 16, 16).cuda()
    #     return dict(x=[x, y])


    # with torch.cuda.device(0):
    #     net = Net(num_feats=32, kernel_size=3, scale=16).cuda()
    #     macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
    #                                              input_constructor=prepare_input,
    #                                              print_per_layer_stat=False, verbose=True)

    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
