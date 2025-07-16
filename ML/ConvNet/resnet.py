import numpy as np
import torch
from torch import nn

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), #output size: 64 112 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x_1 = ResidualBlock(in_channel=64, out_channel=64, s=1)
        self.conv2_x_2 = ResidualBlock(in_channel=64, out_channel=64, s=1)

        self.conv3_x_1 = ResidualBlock(in_channel=64, out_channel=128, s=2)
        self.conv3_x_2 = ResidualBlock(in_channel=128, out_channel=128, s=1)

        self.conv4_x_1 = ResidualBlock(in_channel=128, out_channel=256, s=2)
        self.conv4_x_2 = ResidualBlock(in_channel=256, out_channel=256, s=1)

        self.conv5_x_1 = ResidualBlock(in_channel=256, out_channel=512, s=2)
        self.conv5_x_2 = ResidualBlock(in_channel=512, out_channel=512, s=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)  # Output layer for 1000 classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2_x_1(x)
        x = self.conv2_x_2(x)

        x = self.conv3_x_1(x)
        x = self.conv3_x_2(x)

        x = self.conv4_x_1(x)
        x = self.conv4_x_2(x)

        x = self.conv5_x_1(x)
        x = self.conv5_x_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        return logits
    
class ResNet34(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x_1 = ResidualBlock(in_channel=64, out_channel=64, s=1)
        self.conv2_x_2 = ResidualBlock(in_channel=64, out_channel=64, s=1)
        self.conv2_x_3 = ResidualBlock(in_channel=64, out_channel=64, s=1)

        self.conv3_x_1 = ResidualBlock(in_channel=64, out_channel=128, s=2)
        self.conv3_x_2 = ResidualBlock(in_channel=128, out_channel=128, s=1)
        self.conv3_x_3 = ResidualBlock(in_channel=128, out_channel=128, s=1)
        self.conv3_x_4 = ResidualBlock(in_channel=128, out_channel=128, s=1)

        self.conv4_x_1 = ResidualBlock(in_channel=128, out_channel=256, s=2)
        self.conv4_x_2 = ResidualBlock(in_channel=256, out_channel=256, s=1)
        self.conv4_x_3 = ResidualBlock(in_channel=256, out_channel=256, s=1)
        self.conv4_x_4 = ResidualBlock(in_channel=256, out_channel=256, s=1)
        self.conv4_x_5 = ResidualBlock(in_channel=256, out_channel=256, s=1)
        self.conv4_x_6 = ResidualBlock(in_channel=256, out_channel=256, s=1)

        self.conv5_x_1 = ResidualBlock(in_channel=256, out_channel=512, s=2)
        self.conv5_x_2 = ResidualBlock(in_channel=512, out_channel=512, s=1)
        self.conv5_x_3 = ResidualBlock(in_channel=512, out_channel=512, s=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)  # Output layer for 1000 classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2_x_1(x)
        x = self.conv2_x_2(x)
        x = self.conv2_x_3(x)

        x = self.conv3_x_1(x)
        x = self.conv3_x_2(x)
        x = self.conv3_x_3(x)
        x = self.conv3_x_4(x)

        x = self.conv4_x_1(x)
        x = self.conv4_x_2(x)
        x = self.conv4_x_3(x)
        x = self.conv4_x_4(x)
        x = self.conv4_x_5(x)
        x = self.conv4_x_6(x)

        x = self.conv5_x_1(x)
        x = self.conv5_x_2(x)
        x = self.conv5_x_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        return logits
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, s=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,stride=s, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        
        if s != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out