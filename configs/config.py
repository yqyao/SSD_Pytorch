from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict
import six
import yaml
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import copy
from ast import literal_eval

__C = AttrDict()
cfg = __C

__C.MODEL = AttrDict()

__C.MODEL.NUM_CLASSES = -1
__C.MODEL.TYPE = ''
__C.MODEL.SIZE = '300'
__C.MODEL.CONV_BODY = ''
__C.MODEL.REFINE = False
__C.MODEL.LOAD_PRETRAINED_WEIGHTS = False
__C.MODEL.PRETRAIN_WEIGHTS = ''
__C.MODEL.OBJECT_SCORE = 0.01

__C.TRAIN = AttrDict()
__C.TRAIN.OVERLAP = 0.5
__C.TRAIN.OHEM = True
__C.TRAIN.NEG_RATIO = 3
__C.TRAIN.FOCAL_LOSS = False
__C.TRAIN.FOCAL_LOSS_TYPE = 'SOFTMAX'
__C.TRAIN.BGR_MEAN = [104, 117, 123]
__C.TRAIN.BATCH_SIZE = 1
__C.TRAIN.CHANNEL_SIZE = '48'
__C.TRAIN.WARMUP = True
__C.TRAIN.WARMUP_EPOCH = 2
__C.TRAIN.DEVICE_IDS = [0]
__C.TRAIN.TRAIN_ON = True

__C.SMALL = AttrDict()

__C.SMALL.FEATURE_MAPS = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
__C.SMALL.ARM_CHANNELS = [512, 1024, 512, 256, 256, 256]
__C.SMALL.ODM_CHANNELS = [256, 256, 256, 256]
__C.SMALL.NUM_ANCHORS = [4, 6, 6, 6, 4, 4]
__C.SMALL.STEPS = [[8, 8], [16, 16], [32, 32], [64, 64], [100, 100],
                   [300, 300]]
__C.SMALL.MIN_SIZES = [30, 60, 111, 162, 213, 264]
__C.SMALL.MAX_SIZES = [60, 111, 162, 213, 264, 315]
__C.SMALL.ASPECT_RATIOS = [[2, 0.5], [2, 3, 0.5, 0.333], [2, 3, 0.5, 0.333],
                           [2, 3, 0.5, 0.333], [2, 0.5], [2, 0.5]]
__C.SMALL.VARIANCE = [0.1, 0.2]
__C.SMALL.CLIP = True
__C.SMALL.IMG_WH = [300, 300]
__C.SMALL.INPUT_FIXED = True
__C.SMALL.USE_MAX_SIZE = True

__C.BIG = AttrDict()
__C.BIG.FEATURE_MAPS = [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2],
                        [1, 1]]
__C.BIG.ARM_CHANNELS = [512, 1024, 512, 256, 256, 256, 256]
__C.BIG.ODM_CHANNELS = [256, 256, 256, 256]
__C.BIG.NUM_ANCHORS = [4, 6, 6, 6, 6, 4, 4]
__C.BIG.STEPS = [[8, 8], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256],
                 [512, 512]]
__C.BIG.MIN_SIZES = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
__C.BIG.MAX_SIZES = [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
__C.BIG.ASPECT_RATIOS = [[2, 0.5], [2, 3, 0.5, 0.333], [2, 3, 0.5, 0.333],
                         [2, 3, 0.5, 0.333], [2, 3, 0.5, 0.333], [2, 0.5],
                         [2, 0.5]]
__C.BIG.VARIANCE = [0.1, 0.2]
__C.BIG.CLIP = True
__C.BIG.IMG_WH = [512, 512]
__C.BIG.INPUT_FIXED = True
__C.BIG.USE_MAX_SIZE = True

__C.SOLVER = AttrDict()

__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.BASE_LR = 0.001
__C.SOLVER.GAMMA = 0.1
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.EPOCH_STEPS = []
__C.SOLVER.END_EPOCH = 1
__C.SOLVER.START_EPOCH = 0

__C.DATASETS = AttrDict()

VOCROOT = 'data/datasets/VOCdevkit0712/'
COCOROOT = 'data/datasets/coco2015'

__C.DATASETS.TRAIN_TYPE = []
__C.DATASETS.VAL_TYPE = []
__C.DATASETS.DATAROOT = VOCROOT
__C.DATASETS.DATA_TYPE = ''

__C.DATASETS.SETS = AttrDict()
__C.DATASETS.SETS.VOC = [['0712', '0712_trainval']]
__C.DATASETS.SETS.VOC0712PLUS = [['0712', '0712_trainval_test']]
__C.DATASETS.SETS.VOC0712 = [['2012', '2012_trainval']]
__C.DATASETS.SETS.VOC2007 = [['0712', "2007_test"]]
__C.DATASETS.SETS.COCO = [['2014', 'train'], ['2014', 'valminusminival']]
__C.DATASETS.SETS.COCOval = [['2014', 'minival']]
__C.DATASETS.SETS.VOCROOT = VOCROOT
__C.DATASETS.SETS.COCOROOT = COCOROOT

__C.TEST = AttrDict()
__C.TEST.INPUT_WH = [300, 300]
__C.TEST.CONFIDENCE_THRESH = 0.01
__C.TEST.NMS_TYPE = 'NMS'
__C.TEST.NMS_OVERLAP = 0.45
__C.TEST.BATCH_SIZE = 16

VOC_CLASSES = (
    '__background__',  # always index 0
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

COCO_CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorbike',
                'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key))
    return value_a