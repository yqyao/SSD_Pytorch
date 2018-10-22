# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
from models.model_helper import weights_init
import importlib
from layers.functions.prior_layer import PriorLayer


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        print('Failed to find function: %s', func_name)
        raise


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

    def _init_modules(self):
        self.arm_loc.apply(weights_init)
        self.arm_conf.apply(weights_init)
        if self.cfg.MODEL.REFINE:
            self.odm_loc.apply(weights_init)
            self.odm_conf.apply(weights_init)
        if self.cfg.MODEL.LOAD_PRETRAINED_WEIGHTS:
            weights = torch.load(self.cfg.MODEL.PRETRAIN_WEIGHTS)
            print("load pretrain model {}".format(
                self.cfg.MODEL.PRETRAIN_WEIGHTS))
            if self.cfg.MODEL.TYPE.split('_')[-1] == 'vgg':
                self.extractor.vgg.load_state_dict(weights)
            else:
                self.extractor.load_state_dict(weights, strict=False)

    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.prior_layer = PriorLayer(cfg)
        self.priorbox = PriorBox(cfg)
        self.priors = self.priorbox.forward()
        self.extractor = get_func(cfg.MODEL.CONV_BODY)(self.size,
                                                       cfg.TRAIN.CHANNEL_SIZE)
        if cfg.MODEL.REFINE:
            self.odm_channels = size_cfg.ODM_CHANNELS
            self.arm_num_classes = 2
            self.odm_loc = nn.ModuleList()
            self.odm_conf = nn.ModuleList()
        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()
        self.arm_channels = size_cfg.ARM_CHANNELS
        self.num_anchors = size_cfg.NUM_ANCHORS
        self.input_fixed = size_cfg.INPUT_FIXED
        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()
        for i in range(len(self.arm_channels)):
            if cfg.MODEL.REFINE:
                self.arm_loc += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * 4,
                        kernel_size=3,
                        padding=1)
                ]
                self.arm_conf += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * self.arm_num_classes,
                        kernel_size=3,
                        padding=1)
                ]
                self.odm_loc += [
                    nn.Conv2d(
                        self.odm_channels[i],
                        self.num_anchors[i] * 4,
                        kernel_size=3,
                        padding=1)
                ]
                self.odm_conf += [
                    nn.Conv2d(
                        self.odm_channels[i],
                        self.num_anchors[i] * self.num_classes,
                        kernel_size=3,
                        padding=1)
                ]
            else:
                self.arm_loc += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * 4,
                        kernel_size=3,
                        padding=1)
                ]
                self.arm_conf += [
                    nn.Conv2d(
                        self.arm_channels[i],
                        self.num_anchors[i] * self.num_classes,
                        kernel_size=3,
                        padding=1)
                ]
        if cfg.TRAIN.TRAIN_ON:
            self._init_modules()

    def forward(self, x):

        arm_loc = list()
        arm_conf = list()
        if self.cfg.MODEL.REFINE:
            odm_loc = list()
            odm_conf = list()
            arm_xs, odm_xs = self.extractor(x)
            for (x, l, c) in zip(odm_xs, self.odm_loc, self.odm_conf):
                odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
            odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        else:
            arm_xs = self.extractor(x)
        img_wh = (x.size(3), x.size(2))
        feature_maps_wh = [(t.size(3), t.size(2)) for t in arm_xs]
        for (x, l, c) in zip(arm_xs, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        if self.cfg.MODEL.REFINE:
            output = (arm_loc.view(arm_loc.size(0), -1, 4),
                      arm_conf.view(
                          arm_conf.size(0), -1, self.arm_num_classes),
                      odm_loc.view(odm_loc.size(0), -1, 4),
                      odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                      self.priors if self.input_fixed else self.prior_layer(
                          img_wh, feature_maps_wh))
        else:
            output = (arm_loc.view(arm_loc.size(0), -1, 4),
                      arm_conf.view(arm_conf.size(0), -1, self.num_classes),
                      self.priors if self.input_fixed else self.prior_layer(
                          img_wh, feature_maps_wh))
        return output
