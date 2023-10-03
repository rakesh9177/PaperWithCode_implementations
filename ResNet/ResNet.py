import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input, output, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=output, out_channels=output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output),
        )
        self.downsample = downsample
    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = residual + x
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1)

        self.block1 =nn.Sequential(ResidualBlock(input=64, output=64))
        for _ in range(2):
            self.block1.extend([ResidualBlock(input=64, output=64)])

        self.block2 = nn.Sequential(ResidualBlock(input=64, output=128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1),nn.BatchNorm2d(128))))
        for _ in range(3):
            self.block2.extend([ResidualBlock(input=128, output=128)])

        self.block3 = nn.Sequential(ResidualBlock(input=128, output=256,downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=1),nn.BatchNorm2d(256))))
        for _ in range(4):
            self.block3.extend([ResidualBlock(input=256, output=256)])

        self.block4 = nn.Sequential(ResidualBlock(input=256, output=512,downsample=nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1),nn.BatchNorm2d(512))))
        for _ in range(2):
            self.block4.extend([ResidualBlock(input=512, output=512)])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.input(x)
        for layer in self.block1:
            x = layer(x)
        for layer in self.block2:
            x = layer(x)
        for layer in self.block3:
            x = layer(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


random_data = torch.rand((2, 3, 224, 224))
model = ResNet(1000)
print(model)
output= model(random_data)
print(output.shape)

        


