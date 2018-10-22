# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightSoftmaxLoss(nn.Module):
    def __init__(self, class_num, gamma=2, size_average=True):
        super(WeightSoftmaxLoss, self).__init__()
        # if isinstance(weights, Variable):
        #     self.weights = weights
        # else:
        #     self.weights = Variable(weights)

        self.class_num = class_num
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not weights.is_cuda:
            weights = weights.cuda()
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        weights = weights.view(-1, 1)
        batch_loss = -weights * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss