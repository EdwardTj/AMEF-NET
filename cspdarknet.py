#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/8/29 19:49
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv3d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias, )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FocusReplaceConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, padding, groups=1, bias=False, act="silu"):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padding,
                              groups=groups, bias=bias, )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()

        # 原版
        # expansion=0.5
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)
        # 使用短连接
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        # 原版
        y = self.conv2(self.conv1(x))

        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """
        Spatial pyramid pooling layer used in YOLOv3-SPP
        https://arxiv.org/abs/1406.4729
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool3d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(
            self,
            dep_mul=1.0,
            wid_mul=1.0,
            depthwise=False,
            act="silu",
    ):
        super().__init__()

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # Focus替换为6*6卷积
        self.stem = FocusReplaceConv(1, base_channels, ksize=6, stride=2, padding=2)

        # dark2~dark5中的CSPLayer的n比例1:1:3:1
        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                # bottleneck的个数
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                # bottleneck的个数
                n=base_depth * 3,
                # n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                # bottleneck的个数
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                # bottleneck的个数
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        self.classifi = nn.Sequential(
            nn.Linear(6*6*6*1024,1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,512),
            nn.GroupNorm(64,512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        x = self.dark4(x)
        x = self.dark5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifi(x)
        return x



def main():
    model = CSPDarknet()
    # print(model)
    tmp = torch.randn(1, 1, 192, 192, 192)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()