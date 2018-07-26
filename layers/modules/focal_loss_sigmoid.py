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

    def __init__(self, class_num, alpha=0.25, gamma=2, size_average=True):
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average


    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.sigmoid(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        alpha_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        alpha_mask = Variable(alpha_mask)
        ids = targets.view(-1, 1)
        alpha_mask.scatter_(1, ids.data, self.alpha)
        class_mask.scatter_(1, ids.data, 1.)

        inputs_sign = (inputs >= 0).detach().float().view(-1, 1)

        loss_pos = -1. * torch.pow(1 - P, self.gamma) * torch.log(P) * class_mask * alpha_mask
        loss_neg = torch.pow(P, self.gamma) * ((inputs + torch.log(1 + torch.exp(-inputs))) * inputs_sign + torch.log(1 + torch.exp(inputs)) * (1 - inputs_sign)) * (1 - class_mask) * (1 - alpha_mask)

        batch_loss = loss_neg + loss_pos

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss        