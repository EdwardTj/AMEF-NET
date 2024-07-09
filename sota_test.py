import torch
import torch.nn as nn


class BottleBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_maxpool=False, K=False):
        super().__init__()
        if in_c != out_c * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c * 4)
            )
        else:
            self.downsample = nn.Identity()

        if use_maxpool:
            self.downsample = nn.MaxPool2d(kernel_size=1, stride=2)
            stride = 2

        self.bn1 = nn.BatchNorm2d(in_c)
        self.ac = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=stride)

        self.bn3 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c * 4, kernel_size=1)
        self.K = K

    def forward(self, x):
        print("-------------------")
        print("x:",x.shape)
        out = self.bn1(x)
        out = self.ac(out)
        out = self.conv1(out)
        print("conv1:",out.shape)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.conv2(out)
        print("conv2:", out.shape)

        out = self.bn3(out)
        out = self.ac(out)
        out = self.conv3(out)
        print("conv3:", out.shape)
        x = self.downsample(x)
        print("x:", x.shape)

        return x + out


class resnet_50_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(BottleBlock, [64, 64], 3, use_maxpool_layer=3, K=True)
        self.layer2 = self.make_layer(BottleBlock, [256, 128], 4, use_maxpool_layer=4, K=True)
        self.layer3 = self.make_layer(BottleBlock, [512, 256], 6, use_maxpool_layer=3, K=True)
        self.layer4 = self.make_layer(BottleBlock, [1024, 512], 3, K=True)

        self.bn2 = nn.BatchNorm2d(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 4))

    # 创建一个拼接的模块
    def make_layer(self, module, filters, n_layer, use_maxpool_layer=0, K=False):
        filter1, filter2 = filters
        layers = nn.Sequential()
        layers.add_module('0', module(filter1, filter2, K=K))

        filter1 = filter2 * 4
        for i in range(1, n_layer):
            if i == use_maxpool_layer - 1:
                layers.add_module(str(i), module(filter1, filter2, use_maxpool=True, K=False))
            else:
                layers.add_module(str(i), module(filter1, filter2, K=False))
        return layers

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        return self.fc(out)


model = resnet_50_v2()
model
def main():

    model = resnet_50_v2()
    tmp = torch.randn(1, 3,256,256)
    out = model(tmp)
    # print('resnet:', out.shape)
    # print(model)
    # summary(model, input_size=(1, 182, 218, 182))
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()