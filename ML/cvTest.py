import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.multiprocessing
import datetime
import ConvNet

torch.multiprocessing.freeze_support()

if __name__ == "__main__":
    # model selection code
    model = ConvNet.GoogleNet(10)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
    ])

    epochs = 10
    batch_size = 64
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
    # VGG paper uses SGD with momentum 0.9 and weight decay 0.0005

    print(f"Training {model.__class__.__name__} on CIFAR-10...")

    losses = []
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

        losses.append(epoch_loss / len(trainloader))
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

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = "vgg11_train_test_log.txt"
    with open(log_path, "a") as f:
        f.write(f"\nTrain start time: {now}\n\n")  # 앞에 \n 붙여서 이전 로그와 줄바꿈
        f.write(f"Model:{model.__class__.__name__}\n")
        f.write("Epoch losses: [")
        f.write(", ".join(f"{l:.4f}" for l in losses))
        f.write("]\n\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")