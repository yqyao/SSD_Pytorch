# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightSmoothL1Loss(nn.Module):
    def __init__(self, class_num, size_average=False):
        super(WeightSmoothL1Loss, self).__init__()
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        N = inputs.size(0)
        loc_num = inputs.size(1)
        abs_out = torch.abs(inputs - targets)

        if inputs.is_cuda and not weights.is_cuda:
            weights = weights.cuda()

        weights = weights.view(-1, 1)

        weights = torch.cat((weights, weights, weights, weights), 1)
        mask_big = abs_out >= 1.
        mask_small = abs_out < 1.
        loss_big = weights[mask_big] * (abs_out[mask_big] - 0.5)
        loss_small = weights[mask_small] * 0.5 * torch.pow(
            abs_out[mask_small], 2)
        loss_sum = loss_big.sum() + loss_small.sum()

        if self.size_average:
            loss = loss_sum / N * loc_num
        else:
            loss = loss_sum
        return loss
