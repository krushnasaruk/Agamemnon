import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class BaselineCNN(nn.Module):
    """
    Oversized Baseline CNN for CIFAR-10 classification with dynamic channel support.
    Architecture:
      Conv1: 3 -> channels[0], BatchNorm, ReLU, MaxPool (32x32 -> 16x16)
      Conv2: channels[0] -> channels[1], BatchNorm, ReLU, MaxPool (16x16 -> 8x8)
      Conv3: channels[1] -> channels[2], BatchNorm, ReLU
      Conv4: channels[2] -> channels[3], BatchNorm, ReLU, MaxPool (8x8 -> 4x4)
      Conv5: channels[3] -> channels[4], BatchNorm, ReLU
      AdaptiveAvgPool2d((1, 1))
      Linear: channels[4] -> num_classes
    """

    def __init__(
        self,
        channels: List[int] = [64, 128, 256, 256, 512],
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super(BaselineCNN, self).__init__()
        self.channels = list(channels)
        self.num_classes = num_classes

        # Layer 1
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32 -> 16

        # Layer 2
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channels[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # Layer 3
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels[2])
        self.relu3 = nn.ReLU(inplace=True)

        # Layer 4
        self.conv4 = nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.channels[3])
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 8 -> 4

        # Layer 5
        self.conv5 = nn.Conv2d(self.channels[3], self.channels[4], kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.channels[4])
        self.relu5 = nn.ReLU(inplace=True)

        # Global Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.channels[4], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu2(self.bn2(self.conv2(out))))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.pool4(self.relu4(self.bn4(self.conv4(out))))
        out = self.relu5(self.bn5(self.conv5(out)))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def get_channel_config(self) -> List[int]:
        return list(self.channels)


def build_baseline_cnn(
    channels: Optional[List[int]] = None,
    num_classes: int = 10
) -> BaselineCNN:
    if channels is None:
        channels = [64, 128, 256, 256, 512]
    return BaselineCNN(channels=channels, num_classes=num_classes)
