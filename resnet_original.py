import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetDownBlock(64, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1,1))
        self.classfi = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.GroupNorm(64,256),
            nn.ReLU(),
            # AlexNet采取了DropOut进行正则，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )


    def forward(self, x):
        x = x.to(torch.float32)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.classfi(out)
        return out



def main():

    model = ResNet18()
    tmp = torch.randn(1, 1,182,218,182)
    out = model(tmp)
    # print('resnet:', out.shape)
    # print(model)
    # summary(model, input_size=(1, 182, 218, 182))
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()