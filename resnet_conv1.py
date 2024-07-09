import math

import  torch
from    torch import  nn
from    torch.nn import functional as F


class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=7,
                                    # kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out



class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out,stride=1,):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv3d(ch_in, ch_out, kernel_size=(3,3,3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.conv2 = nn.Conv3d(ch_out, ch_out, kernel_size=(3,3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(

                nn.Conv3d(ch_in, ch_out, kernel_size=(1,1,1), stride=stride),
                nn.BatchNorm3d(ch_out)
            )



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.extra(x) + out
        out = F.relu(out)


        return out




class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),

            DEPTHWISECONV(16, 32),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),

        )
        # 网络的第一层加入注意力机制
        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()
        # followed 4 blocks
        # [b, 16, h, w, L] => [b, 32, h ,w, L]
        self.blk1 = ResBlk(32, 32, stride=1)
        # [b, 32, h, w, L] => [b, 64, h, w, L]
        self.blk2 = ResBlk(32, 64, stride=2)
        # # [b, 64, h, w, L] => [b, 128, h, w, L]
        self.blk3 = ResBlk(64, 128, stride=2)
        # # [b, 128, h, w, L] => [b, 256, h, w, L]
        self.blk4 = ResBlk(128, 256, stride=2)
        # self.blk5 = ResBlk(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1,1))
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=(1,1,1))


        # [b, 256, 7, 7]
        self.outlayer = nn.Linear(256, num_class)
    # def addpool(self,x):
    #     maxpool = self.maxpool
    #     avgpool = self.avgpool
    #     self.pool = (maxpool + avgpool) * 0.5
    #     return self.pool
    def forward(self, x):
        """
        :param x:
        :return:
        """
        # print("x1:",x.shape)
        x=x.to(torch.float32)
        x = F.relu(self.conv1(x))
        # print("conv1:",x.shape)
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        # print("blk1:", x.shape)
        x = self.blk2(x)
        # print("blk2:", x.shape)
        x = self.blk3(x)
        # print("blk3:", x.shape)
        x = self.blk4(x)
        # print("blk4:", x.shape)

        x = self.avgpool(x)

        # print("x.avgpool:",x.shape)
        x = x.view(x.size(0), -1)
        # print("x:--------",x)
        x = self.outlayer(x)
        # print("resnet_x.shape:",x.shape,"---",x)


        return x



def main():
    # blk = ResBlk(64, 128)
    # tmp = torch.randn(1, 64, 25, 25,19)
    # out = blk(tmp)
    # print('block:', out.shape)

    #
    model = ResNet18(2)
    tmp = torch.randn(1, 1,256,256,256)
    out = model(tmp)
    # print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()