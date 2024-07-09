import torch
import torch.nn as nn

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * 2, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 2:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * 2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * 2)
            )

    def forward(self, x):
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # conv3
        out = self.conv3(out)
        out = self.bn3(out)
        # shortcut
        shortcut = self.shortcut(x)
        # merge
        out += shortcut
        out = self.relu(out)
        return out

class ResNeXt50_sota(nn.Module):
    def __init__(self, cardinality=32):
        super(ResNeXt50_sota, self).__init__()
        self.cardinality = cardinality
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64,64,kernel_size=1,stride=1)
        self.layer1 = self.make_layer(ResNeXtBlock, 3, 256, stride=1)
        self.conv3 = nn.Conv3d(512, 512, kernel_size=1, stride=1)
        self.layer2 = self.make_layer(ResNeXtBlock, 4, 512, stride=2)
        self.conv4 = nn.Conv3d(1024, 1024, kernel_size=1, stride=1)
        self.layer3 = self.make_layer(ResNeXtBlock, 6, 1024, stride=2)
        self.conv5 = nn.Conv3d(2048, 2048, kernel_size=1, stride=1)
        self.layer4 = self.make_layer(ResNeXtBlock, 3, 2048, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2)
        )



    def make_layer(self, block, num_blocks, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.cardinality))
        self.in_channels = out_channels * 2
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels, 1, self.cardinality))
            self.in_channels = out_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        # layer1
        out = self.conv2(out)
        out = self.layer1(out)
        # print(out.shape)
        # layer2
        out = self.conv3(out)
        # print(out.shape)
        out = self.layer2(out)
        # layer3
        out = self.conv4(out)
        out = self.layer3(out)
        # layer4
        out = self.conv5(out)
        out = self.layer4(out)
        # print(out.shape)
        # avgpool
        out = self.avgpool(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # fc
        out = self.fc(out)
        return out
def main():

    model = ResNeXt50_sota()
    tmp = torch.randn(2, 1,182,218,182)
    out = model(tmp)
    # print('resnet:', out.shape)
    # print(model)
    # summary(model, input_size=(1, 182, 218, 182))
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()
