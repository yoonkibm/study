"""

"""

import numpy as np
import torch
from torch import nn

class VGG_11(torch.nn.Module):
    def __init__(self, num_class: int = 1000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 64, 112, 112

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 128, 56, 56

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 256, 28, 28

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 512, 14, 14

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output size: 512, 7, 7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)  # Output layer for 1000 classes
        )

    def forward(self, x):
        features = self.encoder(x)
        flattened = torch.flatten(features, 1)
        logits = self.classifier(flattened)
        
        return logits
    
class VGG_13(torch.nn.Module):
    def __init__(self, num_class: int = 1000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 64, 112, 112
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 128, 56, 56
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 256, 28, 28
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 512, 14, 14
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output size: 512, 7, 7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)  # Output layer for 1000 classes
        )

    def forward(self, x):
        features = self.encoder(x)
        flattened = torch.flatten(features, 1)
        logits = self.classifier(flattened)
        
        return logits
    
class VGG_16(torch.nn.Module):
    def __init__(self, num_class: int = 1000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 64, 112, 112

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 128, 56, 56

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 256, 28, 28

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 512, 14, 14

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output size: 512, 7, 7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)  # Output layer for 1000 classes
        )

    def forward(self, x):
        features = self.encoder(x)
        flattened = torch.flatten(features, 1)
        logits = self.classifier(flattened)
        
        return logits
    
class VGG_19(torch.nn.Module):
    def __init__(self, num_class: int = 1000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), #output size: 64, 224, 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 64, 112, 112

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), #output size: 128, 112, 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 128, 56, 56

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), #output size: 256, 56, 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 256, 28, 28

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 28, 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #output size: 512, 14, 14

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), #output size: 512, 14, 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output size: 512, 7, 7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)  # Output layer for 1000 classes
        )

    def forward(self, x):
        features = self.encoder(x)
        flattened = torch.flatten(features, 1)
        logits = self.classifier(flattened)
        
        return logits