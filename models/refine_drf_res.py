# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.dense_conv
from torch.autograd import Variable
from models.model_helper import weights_init


def add_extras(size, in_channel, batch_norm=False):
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


def smooth_conv(size):
    # Extra layers added to resnet for feature scaling
    layers = []
    if size == '300':
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
    else:
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]

    return layers


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class DenseSSDResnet(nn.Module):
    def __init__(self, block, num_blocks, size='300', channel_size='48'):
        super(DenseSSDResnet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.extras = nn.ModuleList(add_extras(str(size), 2048))

        dense_list = models.dense_conv.dense_list_res(channel_size, size)
        self.dense_list0 = nn.ModuleList(dense_list[0])
        self.dense_list1 = nn.ModuleList(dense_list[1])
        self.dense_list2 = nn.ModuleList(dense_list[2])
        self.dense_list3 = nn.ModuleList(dense_list[3])
        self.dense_list4 = nn.ModuleList(dense_list[4])
        self.dense_list5 = nn.ModuleList(dense_list[5])
        self.smooth_list = nn.ModuleList(smooth_conv(str(size)))
        self.smooth1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.dense_list0.apply(weights_init)
        self.dense_list1.apply(weights_init)
        self.dense_list2.apply(weights_init)
        self.dense_list3.apply(weights_init)
        self.dense_list4.apply(weights_init)
        self.dense_list5.apply(weights_init)
        self.smooth_list.apply(weights_init)
        self.smooth1.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Bottom-up
        arm_sources = list()
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)

        c2 = self.layer1(c1)
        dense1_p1 = self.dense_list0[0](c2)
        dense1_p2 = self.dense_list0[1](dense1_p1)
        dense1_p3 = self.dense_list0[2](dense1_p2)
        dense1_p1_conv = self.dense_list0[3](dense1_p1)
        dense1_p2_conv = self.dense_list0[4](dense1_p2)
        dense1_p3_conv = self.dense_list0[5](dense1_p3)

        c3 = self.layer2(c2)
        arm_sources.append(c3)
        dense2_p1 = self.dense_list1[0](c3)
        dense2_p2 = self.dense_list1[1](dense2_p1)
        dense2_p3 = self.dense_list1[2](dense2_p2)
        dense2_p1_conv = self.dense_list1[3](dense2_p1)
        dense2_p2_conv = self.dense_list1[4](dense2_p2)
        dense2_p3_conv = self.dense_list1[5](dense2_p3)

        c4 = self.layer3(c3)
        arm_sources.append(c4)
        dense3_up_conv = self.dense_list2[0](c4)
        dense3_up = self.dense_list2[1](dense3_up_conv)
        dense3_p1 = self.dense_list2[2](c4)
        dense3_p2 = self.dense_list2[3](dense3_p1)
        dense3_p1_conv = self.dense_list2[4](dense3_p1)
        dense3_p2_conv = self.dense_list2[5](dense3_p2)

        c5 = self.layer4(c4)
        c5_ = self.smooth1(c5)
        arm_sources.append(c5_)
        dense4_up1_conv = self.dense_list3[0](c5)
        dense4_up2_conv = self.dense_list3[1](c5)
        dense4_up1 = self.dense_list3[2](dense4_up1_conv)
        dense4_up2 = self.dense_list3[3](dense4_up2_conv)
        dense4_p = self.dense_list3[4](c5)
        dense4_p_conv = self.dense_list3[5](dense4_p)

        c6 = F.relu(self.extras[0](c5), inplace=True)
        c6 = F.relu(self.extras[1](c6), inplace=True)
        arm_sources.append(c6)
        x = c6

        dense5_up1_conv = self.dense_list4[0](c6)
        dense5_up2_conv = self.dense_list4[1](c6)
        dense5_up3_conv = self.dense_list4[2](c6)
        dense5_up1 = self.dense_list4[3](dense5_up1_conv)
        dense5_up2 = self.dense_list4[4](dense5_up2_conv)
        dense5_up3 = self.dense_list4[5](dense5_up3_conv)

        dense_out1 = torch.cat(
            (dense1_p1_conv, c3, dense3_up, dense4_up2, dense5_up3), 1)
        dense_out1 = F.relu(self.dense_list5[0](dense_out1))

        dense_out2 = torch.cat(
            (dense1_p2_conv, dense2_p1_conv, c4, dense4_up1, dense5_up2), 1)
        dense_out2 = F.relu(self.dense_list5[1](dense_out2))

        dense_out3 = torch.cat(
            (dense1_p3_conv, dense2_p2_conv, dense3_p1_conv, c5_, dense5_up1),
            1)
        dense_out3 = F.relu(self.dense_list5[2](dense_out3))

        dense_out4 = torch.cat(
            (dense2_p3_conv, dense3_p2_conv, dense4_p_conv, c6), 1)
        dense_out4 = F.relu(self.dense_list5[3](dense_out4))

        sources = [dense_out1, dense_out2, dense_out3, dense_out4]
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            if k > 1:
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    tmp = x
                    index = k - 3
                    tmp = self.smooth_list[index](tmp)
                    tmp = F.relu(
                        self.smooth_list[index + 1](tmp), inplace=True)
                    arm_sources.append(x)
                    sources.append(tmp)
        return arm_sources, sources


def RefineDRFRes50(size, channel_size='48'):
    return DenseSSDResnet(Bottleneck, [3, 4, 6, 3], size, channel_size)


def RefineDRFRes101(size, channel_size='48'):
    return DenseSSDResnet(Bottleneck, [3, 4, 23, 3], size, channel_size)


def RefineDRFRes152(size, channel_size='48'):
    return DenseSSDResnet(Bottleneck, [3, 8, 36, 3], size, channel_size)
