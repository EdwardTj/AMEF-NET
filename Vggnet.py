import torch
import torch.nn as nn



class VGGNet3D(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet3D, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.features3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.features4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.features5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(64,512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 4096),
        #     # nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        #     # nn.GroupNorm(64, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            # nn.GroupNorm(64,256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features1(x)
        print(x.shape)
        x = self.features2(x)
        print(x.shape)
        x = self.features3(x)
        print(x.shape)
        x = self.features4(x)
        print(x.shape)
        x = self.features5(x)
        print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    model = VGGNet3D(2)
    # print(model)
    tmp = torch.randn(2, 1, 182, 218, 182)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()