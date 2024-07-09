import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


class AlexNet_sota(nn.Module):
    def __init__(self,):
        super(AlexNet_sota, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 48, kernel_size=11, stride=4, padding=2),   
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(48),
            nn.MaxPool3d(kernel_size=3, stride=2),                  
            nn.Conv3d(48, 128, kernel_size=5, padding=2),  
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),                 
            nn.Conv3d(128, 192, kernel_size=3, padding=1),          
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, padding=1),          
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, padding=1),          
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),                 
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 4 * 5*4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),                                  
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
        )
        

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

def main():

    model = AlexNet_sota()
    tmp = torch.randn(2, 1,182,218,182)
    out = model(tmp)
    # print('resnet:', out.shape)
    # print(model)
    # summary(model, input_size=(1, 182, 218, 182))
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()

