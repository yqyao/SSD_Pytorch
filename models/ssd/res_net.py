# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
from models.ssd.resnet_ssd import SSDResnet50
from models.drfssd.init_utils import weights_init
from layers.functions.prior_layer import PriorLayer

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, cfg, phase, size, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.prior_layer = PriorLayer(cfg)
        self.priorbox = PriorBox(cfg)
        self.priors = self.priorbox.forward()
        self.extractor = SSDResnet50(size)
        self.in_channels = cfg['in_channels_res']
        self.num_anchors = cfg['num_anchors']
        self.input_fixed = cfg['input_fixed']
        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()
        for i in range(len(self.in_channels)):
        	self.loc += [nn.Conv2d(self.in_channels[i],
                                self.num_anchors[i]*4, kernel_size=3, padding=1)]
        	self.conf += [nn.Conv2d(self.in_channels[i], self.num_anchors[i]
                                 * self.num_classes, kernel_size=3, padding=1)]

    def forward(self, x):

        loc = list()
        conf = list()
        img_wh = (x.size(3), x.size(2))
        xs = self.extractor(x)
        feature_maps_wh = [(t.size(3), t.size(2)) for t in xs]
        for (x, l, c) in zip(xs, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors if self.input_fixed else self.prior_layer(img_wh, feature_maps_wh)
        )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.extractor.load_state_dict(torch.load(base_file), strict=False)
            print("initing resnet ssd")
            self.extractor.extras.apply(weights_init)
            self.extractor.smooth1.apply(weights_init)
            self.loc.apply(weights_init)
            self.conf.apply(weights_init)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def build_ssd_res(cfg, phase, size=300, num_classes=21, channel_size='48',use_extra_prior=False):
    cfg['use_extra_prior'] = use_extra_prior
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return SSD(cfg, phase, size, num_classes)
