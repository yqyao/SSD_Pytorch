from models.ssd.vgg_net import build_ssd_vgg
from models.ssd.res_net import build_ssd_res
from models.ssd.darknet_net import build_ssd_darknet
from models.drfssd.vgg_drfnet import build_drf_vgg
from models.drfssd.resnet_drfnet import build_drf_res
from models.refine_drfssd.vgg_refine_drfnet import build_refine_drf_vgg
from models.refine_drfssd.resnet_refine_drfnet import build_refine_drf_res
from models.refine_drfssd.vgg_refine_net import build_refine_vgg

net_version = { "ssd_vgg" : build_ssd_vgg,
                "ssd_res" : build_ssd_res,
                "ssd_darknet" : build_ssd_darknet,
                "drf_vgg" : build_drf_vgg,
                "drf_res" : build_drf_res,
                "refine_drf_vgg" : build_refine_drf_vgg,
                "refine_drf_res" : build_refine_drf_res,
                "refine_ssd_vgg" : build_refine_vgg
                }

def model_builder(version, cfg, phase, img_dim, num_classes, channel_size='48'):
    if version not in net_version:
        print("the version is not supported!!!")
        return None
    print("version : ", version)
    net = net_version[version](cfg, phase, img_dim, num_classes, channel_size)
    return net