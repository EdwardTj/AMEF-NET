# 创建AlexNet模型 227*227
import torch
from torch import nn

class changeshape(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in):
        super(changeshape, self).__init__()
        # self.conv1 = nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv3d(ch_in, 48, kernel_size=(1,1,1), stride=1, padding=0)
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=(5,5,5))
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(5,5,5))
        self.bn = nn.BatchNorm3d(48)
        # self.gelu = nn.GELU()
        # self.silu = nn.SiLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x=self.conv2(x)
        out = self.bn(x)
        out = self.relu(out)
        # out = self.silu(out)
        x1 = self.maxpool(out)
        x2 = self.avgpool(out)
        out = (x1 + x2) / 2
        return out

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 特征提取
        self.stage1 = nn.Sequential(
            # 输入通道数为3，因为图片为彩色，三通道
            # 而输出96、卷积核为11*11，步长为4，是由AlexNet模型决定的，后面的都同理
            nn.Conv3d(in_channels=1, out_channels=48, kernel_size=7, stride=2, padding=3),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=128, kernel_size=3, stride=2, padding=1),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=192, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2,return_indices=True,ceil_mode=True)
        self.changechannel = nn.Conv3d(128,48,kernel_size=1,stride=1)
        # self.changeshape = nn.Conv3d(48,48,kernel_size=7,stride=4,padding=1)

        self.upmaxpool = nn.MaxUnpool3d(kernel_size=3,stride=2)
        self.upmaxpool21 = nn.MaxUnpool3d(kernel_size=7, stride=4,padding=1)
        self.add32change = changeshape(128)
        self.add21change = changeshape(48)


        # 全连接层
        self.classifier = nn.Sequential(
            # 全连接的第一层，输入肯定是卷积输出的拉平值，即6*6*256
            # 输出是由AlexNet决定的，为4096
            # nn.AdaptiveAvgPool3d(1),
            # nn.Flatten(start_dim=1),
            nn.Linear(in_features=5 * 5 * 5 * 96, out_features=4096),
            nn.BatchNorm1d(4096),
            # nn.GroupNorm(64,4096),
            nn.ReLU(inplace=True),
            # AlexNet采取了DropOut进行正则，防止过拟合
            nn.Dropout(p=0.6),
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(2048),
            nn.GroupNorm(128,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            nn.GroupNorm(64,1024),
            nn.ReLU(inplace=True),
            # 最后一层，输出1000个类别，也是我们所说的softmax层
            nn.Linear(1024, 2),

        )

    #     self.init_weights()  # 在 __init__ 方法中调用自动初始化函数
    # def init_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
    #             nn.init.kaiming_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # 前向算法
    def forward(self, x):
        # x = self.features(x)
        # print(x.shape)
        x1=self.stage1(x)
        x1,x1_indices = self.maxpool(x1)
        # print("x1:",x1.shape,x1_indices.shape)
        x2 = self.stage2(x1)
        x2,x2_indices = self.maxpool(x2)
        # print("x2:",x2.shape,x2_indices.shape)
        x3 = self.stage3(x2)
        x3,x3_indices = self.maxpool(x3)
        # print("x3:",x3.shape)
        # x3上采样为x2,并与x2相加得到add32,并将add32大小调整为(48,5,5,5)
        x3 = self.upmaxpool(x3,x3_indices)
        # print("x3:",x3.shape)
        add32= x3+x2
        # add32更改通道数和add21一样为48
        add32 = self.add32change(add32)
        # print("add32:",add32.shape)
        # x2上采样为x1
        x2 = self.upmaxpool21(x2,x2_indices)
        # x2更改通道数和x1一样
        x2 = self.changechannel(x2)
        # print("x2:",x2.shape)
        # add21 = x2+x1
        add21 = x2+x1
        add21 = self.add21change(add21)
        # print("add21:",add21.shape)
        # add = add32 concat add21
        add = torch.cat([add32,add21],dim=1)
        # print("add:",add.shape)
        # add classifier

        # 不要忘记在卷积--全连接的过程中，需要将数据拉平，之所以从1开始拉平，是因为我们
        # 批量训练，传入的x为[batch（每批的个数）,x(长),x（宽）,x（通道数）]，因此拉平需要从第1（索引，相当于2）开始拉平
        # 变为[batch,x*x*x]
        out = torch.flatten(add, 1)
        out = self.classifier(out)
        # print("out:", out.shape)
        return out


def main():
    model = AlexNet()
    # print(model)
    tmp = torch.randn(2, 1, 182, 182, 182)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()
