import torch
import torch.nn as nn
from torch.nn import functional as F
# from torchsummary import summary








class Detection_sota(nn.Module):
    def __init__(self):
        super(Detection_sota, self).__init__()
        self.aa = nn.Sequential(
            nn.Conv3d(1,4,kernel_size=3,stride=2,dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(4,4,kernel_size=3,stride=2,dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2,stride=2),
            nn.Conv3d(4,64,kernel_size=3,stride=2,dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2,dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2,stride=2),

            nn.Conv3d(64, 128, kernel_size=2, stride=1, dilation=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=2, stride=1,dilation=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=2, stride=1,dilation=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=2, stride=1,dilation=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=1,stride=1),

            nn.Conv3d(256, 512, kernel_size=1, stride=1,dilation=1),
            # nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=1, stride=1,dilation=1),
            # nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, kernel_size=1, stride=1,dilation=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 2048, kernel_size=1, stride=1,dilation=1),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=1,stride=1),

        )

        self.clssfier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(2048*6*6*6,2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,512),
            nn.Dropout(p=0.5),
            nn.Linear(512,2)

        )
        self._initialize_weights()

    def forward(self, x):
        # x = x.to(torch.float32)
        # print("Detection-----------")
        out = self.aa(x)
        # print(out.shape)
        out = self.clssfier(out)
        return out

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)



def main():

    model = Detection_sota()
    tmp = torch.randn(2, 1,182,218,182)
    out = model(tmp)
    # print('resnet:', out.shape)
    # print(model)
    # summary(model, input_size=(1, 182, 218, 182))
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()