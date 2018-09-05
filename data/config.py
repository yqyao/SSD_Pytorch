# config.py
import os
import os.path


pwd = os.getcwd()
VOCroot = os.path.join(pwd, "data/datasets/VOCdevkit0712/")
COCOroot = os.path.join(pwd, "data/datasets/coco2015")

datasets_dict = {"VOC": [('0712', '0712_trainval')],
            "VOC0712++": [('0712', '0712_trainval_test')],
            "VOC2012" : [('2012', '2012_trainval')],
            "COCO": [('2014', 'train'), ('2014', 'valminusminival')],
            "VOC2007": [('0712', "2007_test")],
            "COCOval": [('2014', 'minival')]}

pretrained_model = {
        "ssd_vgg" : "./weights/pretrained_models/vgg16_reducedfc.pth",
        # "ssd_vgg" : "./weights/pretrained_models/vgg16_best_feature.pth",
        "ssd_res" : "./weights/pretrained_models/resnet101-5d3b4d8f.pth",
        "ssd_darknet" : "./weights/pretrained_models/convert_darknet53.pth",
        "drf_vgg" : "./weights/pretrained_models/vgg16_reducedfc.pth",
        "drf_res" : "./weights/pretrained_models/resnet101-5d3b4d8f.pth",
        "refine_ssd_res" : "./weights/pretrained_models/resnet101-5d3b4d8f.pth",
        "refine_ssd_vgg" : "./weights/pretrained_models/vgg16_reducedfc.pth",
        # "refine_ssd_vgg" : "./weights/pretrained_models/vgg16_best_feature.pth",
        "refine_drf_vgg" : "./weights/pretrained_models/vgg16_reducedfc.pth",
        "refine_drf_res" : "./weights/pretrained_models/resnet101-5d3b4d8f.pth",}

VOC = {
    "300": {
            'feature_maps' : [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1,1)],
            'refine': False,
            'min_dim' : 300,
            'in_channels_vgg': (512, 1024, 512, 256, 256, 256),
            'in_channels_darknet': (256, 512, 1024, 256, 256, 256),
            'in_channels_res': (512, 1024, 512, 256, 256, 256),
            'num_anchors': (4, 6, 6, 6, 4, 4),
            'num_anchors_extra': (6, 6, 6, 6, 4, 4),
            'steps' : [(8, 8), (16, 16), (32, 32), (64, 64), (100, 100), (300, 300)],
            'min_sizes' : [30, 60, 111, 162, 213, 264],
            'max_sizes' : [60, 111, 162, 213, 264, 315],
            'aspect_ratios' : [[2, 1/2], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 1/2], [2, 1/2]],
            'aspect_ratios_extra' : [[2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 1/2], [2, 1/2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'use_extra_prior' : False,
            'num_classes' : 21,
            'img_wh' : (300, 300),
            'input_fixed' : True, # if you want to input different size, you need to set this False.
            "use_max_sizes" : True,
            "epoch_step" : [0, 150, 250],
            "end_epoch" : 250
        },
    "512" : {
            'feature_maps' : [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
            'min_dim' : 512,
            'refine': False,
            'in_channels_vgg': (512, 1024, 512, 256, 256, 256, 256),
            'in_channels_darknet': (256, 512, 1024, 256, 256, 256, 256),
            'in_channels_res': (512, 1024, 512, 256, 256, 256, 256),
            'num_anchors': (4, 6, 6, 6, 6, 4, 4),
            'num_anchors_extra': (6, 6, 6, 6, 6, 4, 4),
            'steps' : [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)],
            'min_sizes'  : [35.84, 76.8, 153.6,  230.4, 307.2, 384.0, 460.8],
            'max_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8,  537.6],
            'aspect_ratios' : [[2, 1/2], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 1/2], [2, 1/2]],
            'aspect_ratios_extra' : [[2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 3, 1/2, 1/3], [2, 1/2], [2, 1/2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'use_extra_prior' : True,
            'num_classes' : 21,
            'img_wh' : (512, 512),
            'input_fixed' : True,
            "use_max_sizes" : True,
            "epoch_step" : [0, 150, 250],
            "end_epoch" : 250
        }
    }

Refine_VOC = {
    "300" : {
            'feature_maps' : [(40, 40), (20, 20), (10, 10), (5, 5)],
            'min_dim' : 320,
            'arm_channels': (512, 1024, 256, 256),
            'odm_channels' : (256, 256, 256, 256),
            'num_anchors': (3, 3, 3, 3),
            'steps' : [(8, 8), (16, 16), (32, 32), (64, 64)],
            'min_sizes' : [32, 64, 128, 256],
            'max_sizes' : [64, 128, 256, 315],
            'aspect_ratios' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'aspect_ratios_extra' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'use_extra_prior' : False,
            'num_classes' : 21,
            'img_wh' : (320, 320),
            'input_fixed' : True,
            "use_max_sizes" : False,
            "epoch_step" : [0, 150, 200],
            "end_epoch" : 240
            },
    "512" : {
            'feature_maps' : [(64, 64), (32, 32), (16, 16), (8, 8)],
            'min_dim' : 512,
            'arm_channels': (512, 1024, 256, 256),
            'odm_channels' : (256, 256, 256, 256),
            'num_anchors': (3, 3, 3, 3, 3),
            'steps' : [(8, 8), (16, 16), (32, 32), (64, 64)],
            'min_sizes' : [32, 64, 128, 256],
            'max_sizes' : [64, 128, 256, 384],
            'aspect_ratios' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'aspect_ratios_extra' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'variance' : [0.1, 0.2],
            'clip' : True,            
            'use_extra_prior' : False,
            'num_classes' : 21,
            'img_wh' : (512, 512),
            'input_fixed' : True,
            "use_max_sizes" : False,
            "epoch_step" : [0, 150, 200],
            "end_epoch" : 250
        }
    }

Refine_COCO = {
    "300" : {
            'feature_maps' : [(40, 40), (20, 20), (10, 10), (5, 5)],
            'min_dim' : 320,
            'arm_channels': (512, 1024, 256, 256),
            'odm_channels' : (256, 256, 256, 256),
            'num_anchors': (3, 3, 3, 3),
            'steps' : [(8, 8), (16, 16), (32, 32), (64, 64)],
            'min_sizes' : [32, 64, 128, 256],
            'max_sizes' : [64, 128, 256, 315],
            'aspect_ratios' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'aspect_ratios_extra' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'use_extra_prior' : False,
            'num_classes' : 81,
            'img_hw' : (320, 320),
            'input_fixed' : True,
            "use_max_sizes" : False,
            "epoch_step" : [0, 90, 120],
            "end_epoch" : 140
            },
    "512" : {
            'feature_maps' : [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)],
            'min_dim' : 512,
            'in_channels_vgg': (512, 1024, 512, 256, 256),
            'num_anchors': (3, 3, 3, 3, 3),
            'steps' : [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)],
            'min_sizes' : [32, 64, 128, 256, 384],
            'max_sizes' : [64, 128, 256, 384, 460],
            'aspect_ratios' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'aspect_ratios_extra' : [[2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2], [2, 1/2]],
            'variance' : [0.1, 0.2],
            'clip' : True,            
            'use_extra_prior' : False,
            'num_classes' : 81,
            'img_wh' : (512, 512),
            'input_fixed' : True,
            "use_max_sizes" : False,
            "epoch_step" : [0, 90, 120],
            "end_epoch" : 140
        }
    }

channels_config = {
    "32_1" : [[32, 16, 16], [32, 16, 16], [32, 32, 16], [32, 16, 16], [32, 16, 16], [608, 1120, 608, 304]],
    "32_2" : [[32, 32, 32], [32, 32, 32], [32, 32, 32], [32, 32, 32], [32, 32, 32], [640, 1152, 640, 352]],
    "48" : [[48, 32, 32], [48, 32, 16], [48, 48, 32], [48, 32, 32], [48, 32, 32], [672, 1184, 672, 336]],
    "64" : [[64, 32, 32], [64, 32, 16], [64, 64, 32], [64, 32, 32], [64, 32, 32], [704, 1216, 704, 336]],
    "96" : [[96, 32, 32], [96, 32, 16], [96, 96, 32], [96, 32, 32], [96, 32, 32], [768, 1280, 768, 336]],
    "128" : [[128, 32, 32], [128, 32, 16], [128, 128, 32], [128, 32, 32], [128, 32, 32], [832, 1344, 832, 336]],
}


COCO = {
    "300" : {
        'feature_maps' : [(38, 38), (19, 38), (10, 10), (5, 5), (3, 3), (1, 1)],
        'min_dim' : 300,
        'refine': False,
        'in_channels_vgg': (512, 1024, 512, 256, 256, 256),
        'in_channels_res': (512, 1024, 512, 256, 256, 256),
        'num_anchors': (4, 6, 6, 6, 4, 4),
        'num_anchors_extra': (6, 6, 6, 6, 4, 4),
        'steps' : [(8, 8), (16, 16), (32, 32), (64, 64), (100, 100), (300, 300)],
        'min_sizes' : [21, 45, 99, 153, 207, 261],
        'max_sizes' : [45, 99, 153, 207, 261, 315],
        'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'aspect_ratios_extra' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance' : [0.1, 0.2],
        'clip' : True,
        'use_extra_prior' : True,
        'num_classes' : 81,
        'img_wh' : (300, 300),
        'input_fixed' : True,
        "use_max_sizes" : True,
        "epoch_step" : [0, 90, 120],
        "end_epoch" : 140
        },
    "512" : {
        'feature_maps' : [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        'min_dim' : 512,
        'refine': False,
        'steps' : [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)],
        'in_channels_vgg': (512, 1024, 512, 256, 256, 256, 256),
        'in_channels_res': (512, 1024, 512, 256, 256, 256, 256),
        'num_anchors': (4, 6, 6, 6, 6, 4, 4),
        'num_anchors_extra': (6, 6, 6, 6, 6, 4, 4),
        'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
        'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
        'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],
        'aspect_ratios_extra' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],
        'variance' : [0.1, 0.2],
        'clip' : True,
        'use_extra_prior' : True,  
        'num_classes' : 81,
        'img_wh' : (512, 512),
        'input_fixed' : True,
        "use_max_sizes" : True,
        "epoch_step" : [0, 90, 120],
        "end_epoch" : 140
        }
    }

cfg_dict = {
    "VOC" : {
            "ssd_vgg" : VOC,
            "ssd_res" : VOC,
            "ssd_darknet" : VOC,
            "drf_vgg" : VOC,
            "drf_res" : VOC,
            "refine_drf_vgg" : VOC,
            "refine_drf_res" : VOC,
            "refine_ssd_vgg" : Refine_VOC,
            "refine_ssd_res" : Refine_VOC,
    },
    "COCO" : {
            "ssd_vgg" : COCO,
            "ssd_res" : COCO,
            "ssd_darknet" : COCO,
            "drf_vgg" : COCO,
            "drf_res" : COCO,
            "refine_drf_vgg" : COCO,
            "refine_drf_res" : COCO,
            "refine_ssd_res" : Refine_COCO,
            "refine_ssd_vgg" : Refine_COCO
    }
}

VOC_CLASSES = (  '__background__',# always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

COCO_CLASSES = ('__background__',
                'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
