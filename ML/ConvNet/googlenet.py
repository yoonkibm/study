import numpy as np
import torch
from torch import nn

class GoogleNet(torch.nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.input_stage = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), #output size: 64 112 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #output size: 64 56 56
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0), #output size: 64 56 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), #output size: 192 56 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # output size: 192 28 28
        )

        self.inception3a = InceptionModule(in_channel=192, ch_1x1=64, ch_3x3_reduce=96, ch_3x3=128, ch_5x5_reduce=16, ch_5x5=32, pool_proj=32)
        self.inception3b = InceptionModule(in_channel=256, ch_1x1=128, ch_3x3_reduce=128, ch_3x3=192, ch_5x5_reduce=32, ch_5x5=96, pool_proj=64)
        
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output size: 480 14 14
        self.inception4a = InceptionModule(in_channel=480, ch_1x1=192, ch_3x3_reduce=96, ch_3x3=208, ch_5x5_reduce=16, ch_5x5=48, pool_proj=64)
        self.inception4b = InceptionModule(in_channel=512, ch_1x1=160, ch_3x3_reduce=112, ch_3x3=224, ch_5x5_reduce=24, ch_5x5=64, pool_proj=64)
        self.inception4c = InceptionModule(in_channel=512, ch_1x1=128, ch_3x3_reduce=128, ch_3x3=256, ch_5x5_reduce=24, ch_5x5=64, pool_proj=64)
        self.inception4d = InceptionModule(in_channel=512, ch_1x1=112, ch_3x3_reduce=144, ch_3x3=288, ch_5x5_reduce=32, ch_5x5=64, pool_proj=64)
        self.inception4e = InceptionModule(in_channel=528, ch_1x1=256, ch_3x3_reduce=160, ch_3x3=320, ch_5x5_reduce=32, ch_5x5=128, pool_proj=128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # output size: 832 7 7

        self.inception5a = InceptionModule(in_channel=832, ch_1x1=256, ch_3x3_reduce=160, ch_3x3=320, ch_5x5_reduce=32, ch_5x5=128, pool_proj=128)
        self.inception5b = InceptionModule(in_channel=832, ch_1x1=384, ch_3x3_reduce=192, ch_3x3=384, ch_5x5_reduce=48,ch_5x5=128, pool_proj=128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output size: 1024 1 1

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),  # Output layer for 1000 classes
        )

    def forward(self, x):
        x = self.input_stage(x)
        x3 = self.inception3a(x)
        x3 = self.inception3b(x3)
        x3 = self.max_pool3(x3)

        x4 = self.inception4a(x3)
        x4 = self.inception4b(x4)
        x4 = self.inception4c(x4)
        x4 = self.inception4d(x4)
        x4 = self.inception4e(x4)
        x4 = self.max_pool4(x4)

        x5 = self.inception5a(x4)
        x5 = self.inception5b(x5)
        x5 = self.avg_pool(x5)
        x5 = torch.flatten(x5, 1)
        logits = self.classifier(x5)

        return logits

class InceptionModule(torch.nn.Module):
    def __init__(self,in_channel: int, ch_1x1: int, ch_3x3_reduce: int, ch_3x3: int, ch_5x5_reduce: int, ch_5x5: int, pool_proj: int):
        super().__init__()

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=ch_1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=ch_3x3_reduce, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_3x3_reduce, out_channels=ch_3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=ch_5x5_reduce, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_5x5_reduce, out_channels=ch_5x5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channel, out_channels=pool_proj, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(x)
        x3 = self.branch5x5(x)
        x4 = self.branch_pool(x)
        outputs = [x1, x2, x3, x4]

        return torch.cat(outputs, 1)