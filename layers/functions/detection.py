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
    def __init__(self, num_classes, bkg_label, cfg, use_arm=False, object_score=0):
        self.num_classes = num_classes
        self.background_label = bkg_label
        #self.thresh = thresh
        self.use_arm = use_arm
        # Parameters used in nms.
        self.variance = cfg['variance']
        self.object_score = object_score

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
        if self.use_arm:
            arm_loc, arm_conf, loc, conf, priors = predictions
            arm_conf = F.softmax(arm_conf.view(-1, 2), 1)
            conf = F.softmax(conf.view(-1, self.num_classes), 1)
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= 0.01 # self.object_score
            conf.data[no_object_index.expand_as(conf.data)] = 0
        else:
            loc, conf, priors = predictions
            conf = F.softmax(conf.view(-1, self.num_classes), 1)  
        loc_data = loc.data
        conf_data = conf.data
        prior_data = priors.data

        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)

        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            self.boxes.expand_(num, self.num_priors, 4)
            self.scores.expand_(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            if self.use_arm:
                default = decode(arm_loc_data[i], prior_data, self.variance)
                default = center_size(default)
                decoded_boxes = decode(loc_data[i], default, self.variance)
            else:
                decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            '''
            c_mask = conf_scores.gt(self.thresh)
            decoded_boxes = decoded_boxes[c_mask]
            conf_scores = conf_scores[c_mask]
            '''

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores