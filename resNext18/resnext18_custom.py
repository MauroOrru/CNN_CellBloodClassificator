import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """
    Un blocco base ResNeXt con convoluzioni a gruppi.
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=16, width=2):
        super(ResNeXtBlock, self).__init__()
        D = width * cardinality  # Numero di filtri effettivi
        
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)

        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNeXt18(nn.Module):
    """
    Modello ResNeXt-18 personalizzato con parametri simili a ResNet-18.
    """
    def __init__(self, num_classes=1000, cardinality=16, width=2):
        super(ResNeXt18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1, cardinality=cardinality, width=width)
        self.layer2 = self._make_layer(128, 2, stride=2, cardinality=cardinality, width=width)
        self.layer3 = self._make_layer(256, 2, stride=2, cardinality=cardinality, width=width)
        self.layer4 = self._make_layer(512, 2, stride=2, cardinality=cardinality, width=width)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, cardinality, width):
        layers = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layers.append(ResNeXtBlock(self.in_channels, out_channels, stride, cardinality, width))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
