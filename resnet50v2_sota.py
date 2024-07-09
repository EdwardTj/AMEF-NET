import torch
import torch.nn as nn



class BottleBlock_set1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_maxpool=False, K=False):
        super().__init__()
        if in_c != out_c * 4:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c * 4, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_c * 4)
            )
        else:
            self.downsample = nn.Identity()

        if use_maxpool:
            self.downsample = nn.MaxPool3d(kernel_size=1, stride=2)
            stride = 2

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.ac = nn.ReLU(inplace=True)


        self.bn2 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, stride=stride)

        self.bn3 = nn.BatchNorm3d(out_c)
        self.conv3 = nn.Conv3d(out_c, out_c * 4, kernel_size=1)
        self.K = K

    def forward(self, x):
        # print("set1:",x.shape)
        out = self.conv1(x)
        # print("set1_conv1:", out.shape)
        out = self.bn1(out)
        out = self.ac(out)

        out = self.conv2(out)
        # print("set1_conv2:", out.shape)
        out = self.bn2(out)
        out = self.ac(out)

        out = self.conv3(out)
        # print("set1_conv3:", out.shape)
        x = self.downsample(x)
        # print("set1_x:", x.shape)
        return x + out



class BottleBlock_set2(nn.Module):
    def __init__(self, in_c, out_c, stride=1, K=False):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_c,out_c,stride=1,kernel_size=1),
            nn.BatchNorm3d(out_c)
        )

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.ac = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.conv3 = nn.Conv3d(out_c, out_c, kernel_size=1)


    def forward(self, x):
        out = self.conv1(x)
        # print("conv1:",out.shape)
        out = self.bn1(out)
        out = self.ac(out)
        # out = self.conv1(out)
        out = self.conv2(out)
        # print("conv2:",out.shape)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.conv3(out)
        # print("conv3:",out.shape)
        x = self.downsample(x)
        # print("set1_x:", x.shape)
        return x + out
        return out


class resnet_50_v2_sota(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.aa1 = nn.Sequential(
            BottleBlock_set1(64,64,stride=2,K=True),
        )
        self.aa2 = nn.Sequential(
            BottleBlock_set2(256,256,stride=1),
            BottleBlock_set2(256,256,stride=1),
        )
        self.aa3 = nn.Sequential(
            BottleBlock_set1(256,512,stride=2,K=True),
        )
        self.aa4=nn.Sequential(
            BottleBlock_set2(2048, 512, stride=1),
            BottleBlock_set2(512, 512, stride=1),
            BottleBlock_set2(512, 512, stride=1),
        )
        self.aa5 = nn.Sequential(
            BottleBlock_set1(512, 1024, stride=2, K=True),
        )
        self.aa6 = nn.Sequential(
            BottleBlock_set2(4096, 1024, stride=1),
            BottleBlock_set2(1024, 1024, stride=1),
            BottleBlock_set2(1024, 1024, stride=1),
            BottleBlock_set2(1024, 1024, stride=1),
            BottleBlock_set2(1024, 1024, stride=1),
        )
        self.aa7 = nn.Sequential(
            BottleBlock_set1(1024, 2048, stride=2, K=True),
        )
        self.aa8 = nn.Sequential(
            BottleBlock_set2(8192, 2048, stride=1),
            BottleBlock_set2(2048, 2048, stride=1),

        )

        # self.layer1 = self.make_layer(BottleBlock_set1, [64, 64], 1, use_maxpool_layer=3, K=True)
        # self.layer2 = self.make_layer(BottleBlock_set2, [256, 256], 2)
        # self.layer3 = self.make_layer(BottleBlock_set1, [256, 128], 1, use_maxpool_layer=3, K=True)
        # self.layer4 = self.make_layer(BottleBlock_set2, [512, 512], 3, K=True)
        # self.layer5 = self.make_layer(BottleBlock_set1, [2048, 1024], 1, use_maxpool_layer=3, K=True)
        # self.layer6 = self.make_layer(BottleBlock_set2, [4096, 2048], 5, K=True)
        # self.layer7 = self.make_layer(BottleBlock_set1, [8192, 4096], 1, use_maxpool_layer=3, K=True)
        # self.layer8 = self.make_layer(BottleBlock_set2, [4096, 2048], 2, K=True)

        self.bn2 = nn.BatchNorm3d(2048)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.7),
            nn.Linear(2048, 2))

    # 创建一个拼接的模块
    # def make_layer(self, module, filters, n_layer, use_maxpool_layer=0, K=False):
    #     filter1, filter2 = filters
    #     layers = nn.Sequential()
    #     layers.add_module('0', module(filter1, filter2, K=K))
    #
    #     filter1 = filter2 * 4
    #     for i in range(1, n_layer):
    #         if i == use_maxpool_layer - 1:
    #             layers.add_module(str(i), module(filter1, filter2, use_maxpool=True, K=False))
    #         else:
    #             layers.add_module(str(i), module(filter1, filter2, K=False))
    #     return layers

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn(out)
        out = self.relu(out)

        out = self.pool1(out)
        out = self.aa1(out)
        # print("aa1:",out.shape)
        out = self.aa2(out)
        # print("aa2:", out.shape)
        out = self.aa3(out)
        # print("aa3:", out.shape)
        out = self.aa4(out)
        # print("aa4:", out.shape)
        out = self.aa5(out)
        # print("aa5:", out.shape)
        out = self.aa6(out)
        # print("aa6:", out.shape)
        out = self.aa7(out)
        # print("aa7:", out.shape)
        out = self.aa8(out)
        # print("aa8:", out.shape)

        # out = self.layer1(out)
        # print("layer1:",out.shape)
        # out = self.layer2(out)
        # print("layer2:", out.shape)
        # out = self.layer3(out)
        # print("layer3:", out.shape)
        # out = self.layer4(out)
        # print("layer4:", out.shape)
        # out = self.layer5(out)
        # print("layer5:", out.shape)
        # out = self.layer6(out)
        # print("layer6:", out.shape)
        # out = self.layer7(out)
        # print("layer7:", out.shape)
        # out = self.layer8(out)
        # print("layer8:", out.shape)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)
        return self.fc(out)


model = resnet_50_v2_sota()
model

def main():

    model = resnet_50_v2_sota()
    tmp = torch.randn(1, 1,182,218,182)
    out = model(tmp)
    # print('resnet:', out.shape)
    # print(model)
    # summary(model, input_size=(1, 182, 218, 182))
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()
