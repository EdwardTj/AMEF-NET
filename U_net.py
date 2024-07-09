import torch
import torch.nn as nn


# 定义3D U-Net模型
import torch
import torch.nn as nn


# UNet模型定义
class UNet3D(nn.Module):
    def __init__(self, ):
        super(UNet3D, self).__init__()
        in_channels=1
        out_channels=2
        # 编码器部分
        self.encoder1 = self._make_encoder_block(in_channels, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)

        # 解码器部分
        self.decoder1 = self._make_decoder_block(512, 256)
        self.decoder2 = self._make_decoder_block(512, 128)
        self.decoder3 = self._make_decoder_block(256, 64)

        # 分类层
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化层
        self.fc = nn.Linear(64, out_channels)  # 全连接层

    def forward(self, x):
        # 编码器部分
        x1 = self.encoder1(x)
        # print("x1:",x1.shape)
        x2 = self.encoder2(x1)
        # print("x2:", x2.shape)
        x3 = self.encoder3(x2)
        # print("x3:", x3.shape)
        x4 = self.encoder4(x3)
        # print("x4:", x4.shape)

        # 解码器部分
        x = self.decoder1(x4)
        # print("x:", x.shape,x3.shape)
        aa=torch.cat((x,x3),dim=1)
        # print("aa:",aa.shape)
        x = self.decoder2(aa)
        # print("x:", x.shape)
        x = self.decoder3(torch.cat([x, x2], dim=1))
        # print("x:", x.shape)

        # 分类层
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return x

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        )


# 创建UNet模型实例

def main():
    model = UNet3D()
    # print(model)
    tmp = torch.randn(2, 1, 224, 224, 224)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()
