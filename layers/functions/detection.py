import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
from utils.box_utils import decode, center_size


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg.MODEL.NUM_CLASSES
        #self.thresh = thresh
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        # Parameters used in nms.
        self.variance = size_cfg.VARIANCE
        self.object_score = cfg.MODEL.OBJECT_SCORE

    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        # loc, conf, priors = predictions
        if self.cfg.MODEL.REFINE:
            arm_loc, arm_conf, loc, conf, priors = predictions
            arm_conf = F.softmax(arm_conf.view(-1, 2), 1)
            conf = F.softmax(conf.view(-1, self.num_classes), 1)
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= self.object_score
            conf.data[no_object_index.expand_as(conf.data)] = 0
        else:
            loc, conf, priors = predictions
            conf = F.softmax(conf.view(-1, self.num_classes), 1)
        loc_data = loc.data
        conf_data = conf.data
        # prior_data = priors.data
        prior_data = priors[:loc_data.size(1), :]

        num = loc_data.size(0)  # batch size

        self.num_priors = prior_data.size(0)

        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes)
        conf_preds = conf_data.view(num, self.num_priors, self.num_classes)
        batch_prior = prior_data.view(-1, self.num_priors, 4).expand(
            (num, self.num_priors, 4))
        batch_prior = batch_prior.contiguous().view(-1, 4)
        if self.cfg.MODEL.REFINE:
            default = decode(
                arm_loc_data.view(-1, 4), batch_prior, self.variance)
            default = center_size(default)
            decoded_boxes = decode(
                loc_data.view(-1, 4), default, self.variance)
        else:
            decoded_boxes = decode(
                loc_data.view(-1, 4), batch_prior, self.variance)

        self.scores = conf_preds.view(num, self.num_priors, self.num_classes)
        self.boxes = decoded_boxes.view(num, self.num_priors, 4)
        return self.boxes, self.scores