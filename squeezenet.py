from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
# from tensorboardX import SummaryWriter
# from torchsummary import summary


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=(1,1,1))
        self.squeeze_activation = nn.ReLU(inplace=True)
        # self.squeeze_activation = nn.GELU()
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes, kernel_size=(1,1,1))
        self.expand1x1_activation = nn.ReLU(inplace=True)
        # self.expand1x1_activation = nn.GELU()
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes, kernel_size=(3,3,3), padding=(1,1,1))
        self.expand3x3_activation = nn.ReLU(inplace=True)
        # self.expand3x3_activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 2, ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2,padding=0),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv3d(64,64,kernel_size=(3,3,3), stride=(2,2,2),padding=(1,1,1))
            # nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2),padding=(1,1,1)),
        )
        self.layer1 = nn.Sequential(
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            # nn.Conv3d(128,128,kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2),padding=(1,1,1)),
        )
        self.layer2 = nn.Sequential(
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2),padding=(1,1,1)),
            # nn.Conv3d(256,256,kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        )
        self.layer3 = nn.Sequential(
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            # nn.Conv3d(384, num_classes, kernel_size=(1, 1, 1)),
            nn.Dropout(p=0.5),
            nn.Conv3d(512, num_classes, kernel_size=(1, 1, 1)),

            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         if m is final_conv:
        #             init.normal_(m.weight, mean=0.0, std=0.01)
        #         else:
        #             init.kaiming_uniform_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        x1= self.layer1(x)
        # print("x1:",x1.shape)
        x2=self.layer2(x1)
        x3=self.layer3(x2)
        out = self.classifier(x3)
        out= torch.flatten(out,1)
        return out
#
# def main():
#
#     model = SqueezeNet()
#     tmp = torch.randn(1, 1,182,218,182)
#     # out = model(tmp)
#     # print('resnet:', out.shape)
#     # print(model)
#     # input = (1, 182, 218, 182)
#     writer = SummaryWriter("pokemon_MRI/logs_seq",tmp)
#     writer.add_graph(model,tmp)
#     writer.close()
#     # writer = summary(model, input_size=(1, 182, 218, 182))
#
#     p = sum(map(lambda p:p.numel(), model.parameters()))
#     print('parameters size:', p)
#
#
# if __name__ == '__main__':
#     main()
