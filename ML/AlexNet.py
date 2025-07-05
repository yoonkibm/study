"""
AlexNet 논문 요약 및 핵심 개념 정리
---

논문: ImageNet Classification with Deep Convolutional Neural Networks (2012)
저자: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton

1. 모델 개요
- ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC 2012)에서 우승
- 기존의 머신러닝 기반 컴퓨터 비전 성능을 압도적으로 초월
- 처음으로 딥러닝(CNN)이 대중적으로 사용되게 만든 계기

2. 아키텍처 구조 (총 8개의 층)
- Conv1: 96 filters, 11x11 kernel, stride 4, ReLU, max-pooling
- Conv2: 256 filters, 5x5 kernel, ReLU, max-pooling
- Conv3: 384 filters, 3x3 kernel, ReLU
- Conv4: 384 filters, 3x3 kernel, ReLU
- Conv5: 256 filters, 3x3 kernel, ReLU, max-pooling
- FC1: 4096 neurons, ReLU, dropout
- FC2: 4096 neurons, ReLU, dropout
- FC3: 1000 neurons (softmax for classification)

3. 주요 기술적 특징
- ReLU 활성화 함수 사용: Sigmoid 대비 학습 속도 빠름
- GPU 병렬처리: 두 개의 GPU에 네트워크 분산 (당시 GPU 메모리 제한 때문)
- Local Response Normalization (LRN): 서로 다른 뉴런의 반응을 정규화하여 일반화 성능 향상 (현대에는 잘 사용되지 않음)
- Dropout: Fully Connected Layer에서 과적합 방지용으로 사용 (p=0.5)
- Data Augmentation: 이미지 회전, 크롭, RGB jittering 등으로 학습 데이터 확장

4. 성능
- ILSVRC 2012 Top-5 Error Rate: 15.3% (2위는 26.2%)
- 1000 클래스 분류에서 압도적인 성능 차이를 보여줌

5. 한계 및 영향
- 커널이 크고 stride가 큼 → 공간 정보 손실
- 학습 속도 개선에는 GPU가 필수였음
- 이후 VGG, GoogLeNet, ResNet 등으로 발전하는 데 큰 기여를 함

요약:
AlexNet은 딥러닝을 컴퓨터 비전에서 주류로 만들었으며, CNN 기반 이미지 분류기의 기초 구조를 확립한 역사적인 모델이다. ReLU, Dropout, GPU 활용, 데이터 증강 등을 실제 적용해 실질적인 성능 개선을 입증한 최초의 사례이다.
"""
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

class AlexNet(torch.nn.Module):
    """
    AlexNet model for image classification.
    This model consists of CNN Layers followed by Fully Connected Layers.
    The architecture is designed to process images.
    """
    def __init__(self, num_class: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # output size: 96, 55, 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #output size: 96, 27, 27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), #output size: 256, 27, 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #output size: 256, 13, 13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), #output size: 384, 13, 13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), #output size: 384, 13, 13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), #output size: 256, 13, 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # output size:256, 6, 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),  # Flatten the output from the conv layers
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)  # Output layer for 1000 classes
        )

    def forward(self, x):
        feature = self.features(x)
        flattenned = torch.flatten(feature, 1)
        logits = self.classifier(flattenned)

        return logits
    
if __name__ == "__main__":
    model = AlexNet(10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor()
    ])

    epochs = 10
    batch_size = 64
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    print("Training AlexNet on CIFAR-10...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
