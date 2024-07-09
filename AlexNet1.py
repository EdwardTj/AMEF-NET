# 创建AlexNet模型 227*227
import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 特征提取
        self.stage1 = nn.Sequential(
            # 输入通道数为3，因为图片为彩色，三通道
            # 而输出96、卷积核为11*11，步长为4，是由AlexNet模型决定的，后面的都同理
            nn.Conv3d(in_channels=1, out_channels=96, kernel_size=7, stride=2,padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2,)
        )

        self.stage2 = nn.Sequential(
            nn.Conv3d(in_channels=96, out_channels=256, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3,stride=2),
        )

        # 全连接层
        self.classifier = nn.Sequential(
            # 全连接的第一层，输入肯定是卷积输出的拉平值，即6*6*256
            # 输出是由AlexNet决定的，为4096
            nn.Linear(in_features=4 * 4 * 4 * 256, out_features=4096),
            nn.BatchNorm1d(4096),
            # nn.GroupNorm(64,4096),
            nn.ReLU(),
            # AlexNet采取了DropOut进行正则，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(2048),
            nn.GroupNorm(64,2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            nn.GroupNorm(64,1024),
            nn.ReLU(),
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
        x1 = self.stage1(x) #(96, 21, 25, 21)
        # print(x1.shape)
        x2 = self.stage2(x1) #(256, 10, 12, 10)
        # print(x2.shape)
        x3 = self.stage3(x2) #(256, 4, 5, 4)
        print(x3.shape)

        # 不要忘记在卷积--全连接的过程中，需要将数据拉平，之所以从1开始拉平，是因为我们
        # 批量训练，传入的x为[batch（每批的个数）,x(长),x（宽）,x（通道数）]，因此拉平需要从第1（索引，相当于2）开始拉平
        # 变为[batch,x*x*x]
        result = torch.flatten(x3, 1)
        result = self.classifier(result)
        # print("result:", result.shape)
        return result


def main():
    model = AlexNet()
    print(model)
    tmp = torch.randn(2, 1, 182, 182, 182)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()
    # m = eca_block(in_channel=32)
    # a = torch.randn((2,32,8,8,8))
    # r = m(a)
    # print(r.shape)