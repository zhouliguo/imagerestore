from torch import Tensor
from torch import nn
import torch
from torchvision import models

class RestoreNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.down = models.resnet50(weights=('pretrained', 'ResNet50_Weights.IMAGENET1K_V1'))
        del self.down.fc
        del self.down.avgpool

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace = True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.down.layer1(x)
        x1 = self.down.layer1(x)
        x2 = self.down.layer2(x1)

        x = self.conv0(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = self.conv5(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = self.conv6(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = self.conv7(x)
        x = self.relu(x)

        x = self.upsample(x)
        x = self.conv8(x)

        return x

# 测试网络
if __name__ == '__main__':
    model = RestoreNet()