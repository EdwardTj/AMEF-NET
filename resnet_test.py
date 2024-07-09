import torch.nn as nn
import torch
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3,3,3), stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            nn.Conv3d(out_ch, out_ch, (3,3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):  # 224x224x3
    # 实现主module:ResNet34
    def __init__(self, num_classes=1):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(1, 64, (7,7,7), stride=2, padding=3, bias=False),  # (224+2*p-)/2(向下取整)+1，size减半->112
            nn.BatchNorm3d(64),  # 112x112x64
            nn.ReLU(inplace=True),
            nn.MaxPool3d((3,3,3), 2, 1)  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64

        # 重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理，但是为了统一操作。。。
        self.layer2 = self.make_layer(64, 128, 4, stride=2)  # 第一个stride=2,剩下3个stride=1;28x28x128
        self.layer3 = self.make_layer(128, 256, 6, stride=2)  # 14x14x256
        self.layer4 = self.make_layer(256, 512, 3, stride=2)  # 7x7x512
        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(  # 首个ResidualBlock需要进行option B处理
            nn.Conv3d(in_ch, out_ch, (1,1,1), stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
            nn.BatchNorm3d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        print(x.shape)
        x = self.pre(x)  # 56x56x64
        # print(x.shape)
        x = self.layer1(x)  # 56x56x64
        # print(x.shape)
        x = self.layer2(x)  # 28x28x128
        # print(x.shape)
        x = self.layer3(x)  # 14x14x256
        # print(x.shape)
        x = self.layer4(x)  # 7x7x512
        # print(x.shape)
        x = F.avg_pool3d(x, (6,7,6))  # 1x1x512
        print(x.shape)
        x = x.view(x.size(0), -1)  # 将输出拉伸为一行：1x512
        x = self.fc(x)  # 1x1
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        return nn.Sigmoid()(x)  # 1x1，将结果化为(0~1)之间


def main():
    # blk = ResBlk(64, 128)
    # tmp = torch.randn(1, 64, 25, 25,19)
    # out = blk(tmp)
    # print('block:', out.shape)

    #
    model = ResNet34(2)
    tmp = torch.randn(1, 1,182,218,182)
    out = model(tmp)
    # print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()