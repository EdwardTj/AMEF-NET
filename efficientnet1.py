import math

import torch
import torch.nn as nn

# 转置卷积上采样
class Upsample(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in,ch_out):
        super(Upsample, self).__init__()
        # self.conv1 = nn.Conv3d(ch_in, ch_out, kernel_size=(1,1,1), stride=1, padding=0)
        # self.bn1 = nn.BatchNorm3d(ch_out)
        self.conv2 = nn.ConvTranspose3d(ch_in, ch_out, kernel_size=(2,2,2), stride=(2,2,2))
        self.bn2 = nn.BatchNorm3d(ch_out)
        # self.gelu = nn.GELU()
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.silu(out)

        return out

class changeshape(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in,ch_out):
        super(changeshape, self).__init__()
        # self.conv1 = nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=2,padding=1)
        # self.conv2 = nn.Conv3d(ch_in, 112, kernel_size=(1,1,1), stride=1, padding=0)
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=(6,6,6))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(6,6,6))
        self.bn = nn.BatchNorm3d(ch_out)
        # self.gelu = nn.GELU()
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        # x=self.conv2(x)
        out = self.bn(x)
        # out = self.gelu(out)
        out = self.silu(out)
        x1 = self.maxpool(out)
        x2 = self.avgpool(out)
        out = (x1 + x2) / 2
        return out

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups: int = 1,
                 activation: nn.Module = nn.Identity()):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.conv(x)

# SEModule
class SEModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels, out_features=int(in_channels // reduction), bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=int(in_channels // reduction), out_features=in_channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w ,d= x.shape
        out = self.conv(x).view(b, c, 1, 1,1)
        return x * out.expand_as(x)




class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, channels_factor, downsample: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tmp_channels = self.in_channels * channels_factor
        self.stride = 1 if not downsample else 2
        self.conv = nn.Sequential(
            ConvBNSiLU(in_channels=self.in_channels, out_channels=self.tmp_channels, kernel_size=1, stride=1, padding=0,
                       activation=nn.SiLU(inplace=True)),
            ConvBNSiLU(in_channels=self.tmp_channels, out_channels=self.tmp_channels, kernel_size=3, stride=self.stride,
                       padding=1, groups=self.tmp_channels, activation=nn.SiLU(inplace=True)),
            SEModule(in_channels=self.tmp_channels, reduction=4),
            ConvBNSiLU(in_channels=self.tmp_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                       padding=0),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.in_channels == self.out_channels: return out + x
        return out


class EfficientNetB0(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        CFG = [1, 2, 2, 3, 3, 4, 1]
        self.conv = ConvBNSiLU(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.mbconv1_1 = self._make_layers(CFG[0], in_channels=32, out_channels=16, channels_factor=1, downsample=False)
        self.mbconv6_2 = self._make_layers(CFG[1], in_channels=16, out_channels=24, channels_factor=6, downsample=True)
        self.mbconv6_3 = self._make_layers(CFG[2], in_channels=24, out_channels=40, channels_factor=6, downsample=True)
        self.mbconv6_4 = self._make_layers(CFG[3], in_channels=40, out_channels=80, channels_factor=6, downsample=True)
        self.mbconv6_5 = self._make_layers(CFG[4], in_channels=80, out_channels=112, channels_factor=6,
                                           downsample=False)
        self.mbconv6_6 = self._make_layers(CFG[5], in_channels=112, out_channels=192, channels_factor=6,
                                           downsample=True)
        self.mbconv6_7 = self._make_layers(CFG[6], in_channels=192, out_channels=320, channels_factor=6,
                                           downsample=False)
        self.fc = nn.Sequential(
            ConvBNSiLU(in_channels=496, out_channels=1280, kernel_size=1, stride=1, padding=0,
                       activation=nn.SiLU(inplace=True)),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features= 1280, out_features=512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(2048),
            nn.GroupNorm(64, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(256, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=128, out_features=n_classes, bias=True)
        )
        self.x7convx5 = Upsample(320,112)
        self.x5convx3 = Upsample(112,40)
        self.x3convx2 = Upsample(40,24)
        self.add75change = changeshape(112,112)
        self.add53change = changeshape(40,40)
        self.add32change = changeshape(24,24)


    def _make_layers(self, num_layers, in_channels, out_channels, channels_factor, downsample):
        layers = [MBConv(in_channels=in_channels, out_channels=out_channels, channels_factor=channels_factor,
                         downsample=downsample)]
        for _ in range(num_layers - 1):
            layers.append(MBConv(in_channels=out_channels, out_channels=out_channels, channels_factor=channels_factor))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("x:",x.shape)
        x = self.conv(x)
        # print('# Conv output shape:', x.shape)
        x = self.mbconv1_1(x)
        # print('# MBConv1_1 output shape:', x.shape)
        x2 = self.mbconv6_2(x)
        # print('# MBConv6_2 output shape:', x2.shape)
        x3 = self.mbconv6_3(x2)
        # print('# MBConv6_3 output shape:', x3.shape)
        x4 = self.mbconv6_4(x3)
        # print('# MBConv6_4 output shape:', x4.shape)
        x5 = self.mbconv6_5(x4)
        # print('# MBConv6_5 output shape:', x5.shape)
        x6 = self.mbconv6_6(x5)
        # print('# MBConv6_6 output shape:', x6.shape)
        x7 = self.mbconv6_7(x6)
        # print('# MBConv6_7 output shape:', x7.shape)
        # x7反卷积为x5后与x5相加，得到add75,将x75大小调整为(48,6,6,6)
        x77 = self.x7convx5(x7)
        # print("x77:",x77.shape)
        add75 = x77+x5
        add75 = self.add75change(add75)
        # print("add75:",add75.shape)
        # x5反卷积为x3后与x3相加，得到add53,将x53大小调整为(48,6,6,6)
        x5 = self.x5convx3(x5)
        add53 = x5 + x3
        add53 = self.add53change(add53)
        # print("add53:", add53.shape)
        # x3反卷积为x2后与x2相加得到add32,将x32大小调整为(48,6,6,6)
        x3 = self.x3convx2(x3)
        # print("x3:",x3.shape)
        add32 = x3 + x2
        add32 = self.add32change(add32)
        # print("add32:", add32.shape)
        out = torch.cat((add75,add53,add32,x7),dim=1)
        # print("out:",out.shape)
        out = self.fc(out)
        # print('# FC output shape:', x.shape)
        return out

# def main():
#     model = EfficientNetB0(1,2)
#     # print(model)
#     tmp = torch.randn(2, 1, 192, 192, 192)
#     out = model(tmp)
#     print('resnet:', out.shape)
#
#     p = sum(map(lambda p: p.numel(), model.parameters()))
#     print('parameters size:', p)
#
#
# if __name__ == '__main__':
#     main()
    # m = eca_block(in_channel=32)
    # a = torch.randn((2,32,8,8,8))
    # r = m(a)
    # print(r.shape)

# inputs = torch.randn(4, 3, 224, 224)
# cnn = EfficientNetB0(in_channels=3, n_classes=1000)
# outputs = cnn(inputs)
import torch
from thop import profile
from thop import clever_format
def main():
    model = EfficientNetB0(1,2)
    # print(model)
    # tmp = torch.randn(2, 1, 192, 192, 192)
    # out = model(tmp)
    # print('resnet:', out.shape)
    input_size = (2, 1, 192, 192, 192)
    input_data = torch.randn(input_size)
    flops, params = profile(model, inputs=(input_data,))
    flops = clever_format(flops, "%.3f")
    print(f"FLOPs: {flops}","parameter:",params)
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)

if __name__ == '__main__':
    main()