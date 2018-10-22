# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLossSigmoid(nn.Module):
    '''
    sigmoid version focal loss
    '''

    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.sigmoid(inputs)
        alpha_mask = self.alpha * targets
        loss_pos = -1. * torch.pow(
            1 - P, self.gamma) * torch.log(P) * targets * alpha_mask
        loss_neg = -1. * torch.pow(1 - P, self.gamma) * torch.log(1 - P) * (
            1 - targets) * (1 - alpha_mask)
        batch_loss = loss_neg + loss_pos
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
