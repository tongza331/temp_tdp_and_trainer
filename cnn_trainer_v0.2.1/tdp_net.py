from importTDP_lib import *

class TdpNet(nn.Module):
    def __init__(self, num_classes):
        super(TdpNet, self).__init__()

    ## input is 3x224x224
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 64x224x224
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64x224x224
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x112x112
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128x112x112
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 128x112x112
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x56x56
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 256x56x56
            nn.ELU(inplace=True), 
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 256x56x56
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256x28x28
        )

        self.classifier = nn.Sequential(
            nn.Linear(28*28*256, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 28*28*256)
        x = self.classifier(x)
        return x