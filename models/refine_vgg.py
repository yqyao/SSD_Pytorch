# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from models.model_helper import FpnAdapter, WeaveAdapter, weights_init


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


def add_extras(size):
    layers = []
    layers += [nn.Conv2d(1024, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    return layers


# def last_layer_trans():
#     return nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#                   nn.ReLU(inplace=True),
#                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

# def trans_layers(size):
#     layers = list()
#     layers += [nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1,           padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))]
#     layers += [nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1,           padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))]
#     layers += [nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1,           padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))]

#     return layers

# def latent_layers(size):
#     layers = []
#     for i in range(3):
#         layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
#     return layers

# def up_layers(size):
#     layers = []
#     for i in range(3):
#         layers += [nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)]
#     return layers


class VGG16Extractor(nn.Module):
    def __init__(self, size, channel_size='48'):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.extras = nn.ModuleList(add_extras(str(size)))
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(1024, 8)
        # self.last_layer_trans = last_layer_trans()
        # self.trans_layers = nn.ModuleList(trans_layers(str(size)))
        # self.latent_layers = nn.ModuleList(latent_layers((str(size))))
        # self.up_layers = nn.ModuleList(up_layers(str(size)))
        self.fpn = FpnAdapter([512, 1024, 256, 256], 4)
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        # self.last_layer_trans.apply(weights_init)
        # self.trans_layers.apply(weights_init)
        # self.latent_layers.apply(weights_init)
        # self.up_layers.apply(weights_init)

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

        for i in range(23):
            x = self.vgg[i](x)
        #38x38
        c2 = x
        c2 = self.L2Norm_4_3(c2)
        arm_sources.append(c2)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        #19x19
        c3 = x
        c3 = self.L2Norm_5_3(c3)
        arm_sources.append(c3)

        # 10x10
        x = F.relu(self.extras[0](x), inplace=True)
        x = F.relu(self.extras[1](x), inplace=True)
        c4 = x
        arm_sources.append(c4)

        # 5x5
        x = F.relu(self.extras[2](x), inplace=True)
        x = F.relu(self.extras[3](x), inplace=True)
        c5 = x
        arm_sources.append(c5)

        if len(self.extras) > 4:
            x = F.relu(self.extras[4](x), inplace=True)
            x = F.relu(self.extras[5](x), inplace=True)
            c6 = x
            arm_sources.append(c6)

        # x = self.last_layer_trans(x)
        # odm_sources.append(x)

        # trans_layer_list = list()

        # for(p, t) in zip(arm_sources, self.trans_layers):
        #     trans_layer_list.append(t(p))

        # trans_layer_list.reverse()
        # for (t, u, l) in zip(trans_layer_list, self.up_layers, self.latent_layers):
        #     x = F.relu(l(F.relu(u(x)+ t, inplace=True)), inplace=True)
        #     odm_sources.append(x)

        # odm_sources.reverse()
        odm_sources = self.fpn(arm_sources)
        return arm_sources, odm_sources


def refine_vgg(size, channel_size='48'):
    return VGG16Extractor(size)