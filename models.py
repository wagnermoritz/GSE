import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F


class CAMNet(nn.Module):
    '''
    Class for the last fully connected layer of CNNs used for computing the
    class activation map.
    '''
    def __init__(self, numclasses=1000, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, numclasses, bias=False)

    def forward(self, x):
        sh = x.shape
        x = x.view(*sh[:2], sh[2] * sh[3]).mean(-1).view(sh[0], -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def getBasicCNN():
    return nn.Sequential(nn.Conv2d(3, 64, 3),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, 3),
                         nn.ReLU(),
                         nn.MaxPool2d(2),
                         nn.Conv2d(64, 128, 3),
                         nn.ReLU(),
                         nn.Conv2d(128, 128, 3),
                         nn.ReLU(),
                         nn.MaxPool2d(2),
                         nn.Flatten(),
                         nn.Linear(3200, 256),
                         nn.Linear(256, 10))
    

class ResBlock(nn.Module):
    def __init__(self, ins, outs, stride=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(ins, outs, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outs)
        self.conv2 = nn.Conv2d(outs, outs, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outs)
        if stride != 1 or ins != outs:
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, outs//4, outs//4), "constant", 0)
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        res += self.shortcut(x)
        res = self.relu(res)
        return res
    

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.blockLayer(16, 16, 1)
        self.layer2 = self.blockLayer(16, 32, 2)
        self.layer3 = self.blockLayer(32, 64, 2)
        self.linear = nn.Linear(64, num_classes)


    def blockLayer(self, ins, outs, stride):
        return nn.Sequential(
            ResBlock(ins, outs, stride),
            ResBlock(outs, outs, 1),
            ResBlock(outs, outs, 1))
    

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = F.avg_pool2d(res, res.shape[3])
        res = res.view(res.shape[0], -1)
        res = self.linear(res)
        return res
    

class WideBlock(nn.Module):
    def __init__(self, ins, outs, stride, drop):
        super().__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(ins)
        self.conv1 = nn.Conv2d(ins, outs, 3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=drop)
        self.bn2 = nn.BatchNorm2d(outs)
        self.conv2 = nn.Conv2d(outs, outs, 3, stride=stride, padding=1, bias=True)

        if stride != 1 or ins != outs:
            self.shortcut = nn.Conv2d(ins, outs, 1, stride=stride, bias=True)
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        res = self.conv1(self.relu(self.bn1(x)))
        res = self.dropout(res)
        res = self.conv2(self.relu(self.bn2(res)))
        res += self.shortcut(x)
        return res


class WideResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.layer1 = self.blockLayer(16, 160, 1, 0.3)
        self.layer2 = self.blockLayer(160, 320, 2, 0.3)
        self.layer3 = self.blockLayer(320, 640, 2, 0.3)
        self.bn1 = nn.BatchNorm2d(640)
        self.linear = nn.Linear(640, num_classes)

    def blockLayer(self, ins, outs, stride, drop):
        return nn.Sequential(
            WideBlock(ins, outs, stride, drop),
            WideBlock(outs, outs, 1, drop),
            WideBlock(outs, outs, 1, drop),
            WideBlock(outs, outs, 1, drop))

    def forward(self, x):
        res = self.conv1(x)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.relu(self.bn1(res))
        res = F.avg_pool2d(res, 8)
        res = res.view(res.shape[0], -1)
        res = self.linear(res)
        return res