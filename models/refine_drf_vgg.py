# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from models.model_helper import weights_init
import models.refine_dense_conv


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            x) * x
        return out


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7,
        nn.ReLU(inplace=True)
    ]
    return layers


extras_cfg = {
    # '300': [256, 'S', 512, 128, 'S', 256],
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [
        256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S',
        256
    ],
}

base = {
    '300': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
    '512': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
}


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [
                    nn.Conv2d(
                        in_channels,
                        cfg[k + 1],
                        kernel_size=(1, 3)[flag],
                        stride=2,
                        padding=1)
                ]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
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


class VGG16Extractor(nn.Module):
    def __init__(self, size, channel_size='48'):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.extras = nn.ModuleList(add_extras(extras_cfg[str(size)], 1024))
        self.L2Norm1 = L2Norm(256, 20)
        self.L2Norm = L2Norm(512, 20)
        self.L2Norm3 = L2Norm(1024, 10)
        self.L2Norm4 = L2Norm(512, 10)
        self.L2Norm5 = L2Norm(256, 10)
        dense_list = models.refine_dense_conv.dense_list_vgg(
            channel_size, str(size))
        self.dense_list0 = nn.ModuleList(dense_list[0])
        self.dense_list1 = nn.ModuleList(dense_list[1])
        self.dense_list2 = nn.ModuleList(dense_list[2])
        self.dense_list3 = nn.ModuleList(dense_list[3])
        self.dense_list4 = nn.ModuleList(dense_list[4])
        self.dense_list5 = nn.ModuleList(dense_list[5])
        self.smooth_list = nn.ModuleList(smooth_conv(str(size)))
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

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources = list()
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu

        for k in range(16):
            x = self.vgg[k](x)

        # 75x75
        dense1 = x
        dense1 = self.L2Norm1(dense1)
        dense1_p1 = self.dense_list0[0](dense1)
        dense1_p2 = self.dense_list0[1](dense1_p1)
        dense1_p3 = self.dense_list0[2](dense1_p2)
        dense1_p1_conv = self.dense_list0[3](dense1_p1)
        dense1_p2_conv = self.dense_list0[4](dense1_p2)
        dense1_p3_conv = self.dense_list0[5](dense1_p3)
        # p = self.add_conv[1](p)

        for k in range(16, 23):
            x = self.vgg[k](x)
        #38x38
        dense2 = x
        dense2 = self.L2Norm(dense2)
        arm_sources.append(dense2)
        dense2 = F.relu(self.dense_list1[0](dense2), inplace=True)
        dense2_p1 = self.dense_list1[1](dense2)
        dense2_p2 = self.dense_list1[2](dense2_p1)
        dense2_p3 = self.dense_list1[3](dense2_p2)
        dense2_p1_conv = self.dense_list1[4](dense2_p1)
        dense2_p2_conv = self.dense_list1[5](dense2_p2)
        dense2_p3_conv = self.dense_list1[6](dense2_p3)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        #19x19
        dense3 = x
        arm_sources.append(dense3)
        dense3 = self.L2Norm3(dense3)
        dense3 = F.relu(self.dense_list2[0](dense3), inplace=True)
        dense3_up_conv = self.dense_list2[1](dense3)
        dense3_up = self.dense_list2[2](dense3_up_conv)
        dense3_p1 = self.dense_list2[3](dense3)
        dense3_p2 = self.dense_list2[4](dense3_p1)
        dense3_p1_conv = self.dense_list2[5](dense3_p1)
        dense3_p2_conv = self.dense_list2[6](dense3_p2)

        x = F.relu(self.extras[0](x), inplace=True)
        x = F.relu(self.extras[1](x), inplace=True)

        #10x10
        dense4 = x
        arm_sources.append(dense4)
        dense4 = self.L2Norm4(dense4)
        dense4 = F.relu(self.dense_list3[0](dense4), inplace=True)
        dense4_up1_conv = self.dense_list3[1](dense4)
        dense4_up2_conv = self.dense_list3[2](dense4)
        dense4_up1 = self.dense_list3[3](dense4_up1_conv)
        dense4_up2 = self.dense_list3[4](dense4_up2_conv)
        dense4_p = self.dense_list3[5](dense4)
        dense4_p_conv = self.dense_list3[6](dense4_p)

        x = F.relu(self.extras[2](x), inplace=True)
        x = F.relu(self.extras[3](x), inplace=True)

        #5x5
        dense5 = x
        arm_sources.append(dense5)
        dense5 = self.L2Norm5(dense5)
        dense5 = F.relu(self.dense_list4[0](dense5), inplace=True)
        dense5_up1_conv = self.dense_list4[1](dense5)
        dense5_up2_conv = self.dense_list4[2](dense5)
        dense5_up3_conv = self.dense_list4[3](dense5)
        dense5_up1 = self.dense_list4[4](dense5_up1_conv)
        dense5_up2 = self.dense_list4[5](dense5_up2_conv)
        dense5_up3 = self.dense_list4[6](dense5_up3_conv)

        dense_out1 = torch.cat(
            (dense1_p1_conv, dense2, dense3_up, dense4_up2, dense5_up3), 1)
        dense_out1 = F.relu(self.dense_list5[0](dense_out1))
        sources.append(dense_out1)

        dense_out2 = torch.cat(
            (dense1_p2_conv, dense2_p1_conv, dense3, dense4_up1, dense5_up2),
            1)
        dense_out2 = F.relu(self.dense_list5[1](dense_out2))
        sources.append(dense_out2)

        dense_out3 = torch.cat((dense1_p3_conv, dense2_p2_conv, dense3_p1_conv,
                                dense4, dense5_up1), 1)
        dense_out3 = F.relu(self.dense_list5[2](dense_out3))
        sources.append(dense_out3)

        dense_out4 = torch.cat(
            (dense2_p3_conv, dense3_p2_conv, dense4_p_conv, dense5), 1)
        dense_out4 = F.relu(self.dense_list5[3](dense_out4))
        sources.append(dense_out4)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            if k > 3:
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    tmp = x
                    index = k - 5
                    tmp = self.smooth_list[index](tmp)
                    tmp = F.relu(
                        self.smooth_list[index + 1](tmp), inplace=True)
                    arm_sources.append(x)
                    sources.append(tmp)
        return arm_sources, sources


def RefineDRFVgg(size, channel_size='48'):
    return VGG16Extractor(size, channel_size)
