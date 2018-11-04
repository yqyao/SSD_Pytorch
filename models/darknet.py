# -*- coding: utf-8 -*-
# Written by yq_yao
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_helper import weights_init


def add_extras(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    if size == '300':
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)]
    else:
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    return layers


class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01, eps=1e-05, affine=True)

    def forward(self, x):
        return F.leaky_relu(
            self.bn(self.conv(x)), negative_slope=0.1, inplace=True)


class DarknetBlock(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in // 2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class Darknet19(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.layer5 = self._make_layer5()
        self.extras = nn.ModuleList(add_extras(str(size), 1024))

    def _make_layer1(self):
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBN(32, 64, kernel_size=3, stride=1, padding=1)
        ]
        return nn.Sequential(*layers)

    def _make_layer2(self):
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBN(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBN(128, 64, kernel_size=1, stride=1),
            ConvBN(64, 128, kernel_size=3, stride=1, padding=1)
        ]
        return nn.Sequential(*layers)

    def _make_layer3(self):
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ConvBN(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBN(256, 128, kernel_size=1, stride=1),
            ConvBN(128, 256, kernel_size=3, stride=1, padding=1)
        ]
        return nn.Sequential(*layers)

    def _make_layer4(self):
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ConvBN(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBN(512, 256, kernel_size=1, stride=1),
            ConvBN(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBN(512, 256, kernel_size=1, stride=1),
            ConvBN(256, 512, kernel_size=3, stride=1, padding=1)
        ]
        return nn.Sequential(*layers)

    def _make_layer5(self):
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ConvBN(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBN(1024, 512, kernel_size=1, stride=1),
            ConvBN(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBN(1024, 512, kernel_size=1, stride=1),
            ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        sources = [c3, c4, c5]
        x = c5
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


class Darknet53(nn.Module):
    def __init__(self, num_blocks, size):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[4], stride=2)
        self.extras = nn.ModuleList(add_extras(str(size), 1024))
        self._init_modules()

    def _make_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in, ch_in * 2, stride=stride, padding=1)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        sources = [c3, c4, c5]
        x = c5
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


def SSDarknet53(size, channel_size='48'):
    return Darknet53([1, 2, 8, 8, 4], size)


def SSDarknet19(size, channel_size='48'):
    return Darknet19(size)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model3 = SSDarknet19(size=300)
    with torch.no_grad():
        model3.eval()
        x = torch.randn(16, 3, 300, 300)
        model3.cuda()
        model3(x.cuda())
        import time
        st = time.time()
        for i in range(100):
            model3(x.cuda())
        print(time.time() - st)
