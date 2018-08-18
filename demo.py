import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import pickle
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC, COCO, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, VOC_CLASSES, preproc, model_builder, pretrained_model, COCO_CLASSES, VOC_CLASSES, datasets_dict, cfg_dict
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from utils.box_utils import draw_rects
import numpy as np
from utils.timer import Timer
import time
import os 
import sys
import cv2

def arg_parse():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')

    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",default = "images", type = str)
    parser.add_argument('--weights', default='weights/ssd_vggepoch_250_300.pth',type=str, help='Trained state_dict file path to open')

    parser.add_argument('-v', '--version', default='ssd_vgg',
                        help='dense_ssd or origin_ssd version.')
    parser.add_argument('-s', '--size', default='300',
                        help='300 or 512 input size.')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC ,VOC0712++ or COCO dataset')
    parser.add_argument('-c', '--channel_size', default='48',
                        help='channel_size 32_1, 32_2, 48, 64, 96, 128')
    parser.add_argument('--save_folder', default='output/', type=str,
                        help='File path to save results')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', default=200, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, help='Use cuda to train model')

    args = parser.parse_args()
    return args


def im_detect(img, net, detector, cfg, transform, thresh=0.01):
    with torch.no_grad():
        t0 = time.time()
        w, h = img.shape[1], img.shape[0]
        x = transform(img)[0].unsqueeze(0)
        x = x.cuda()
        t1 = time.time()
        output = net(x)
        boxes, scores = detector.forward(output)  
        t2 = time.time()
        max_conf, max_id = scores[0].topk(1, 1, True, True)
        pos = max_id > 0
        if len(pos) == 0:
            return np.empty((0,6))
        boxes = boxes[0][pos.view(-1, 1).expand(len(pos), 4)].view(-1, 4)
        scores = max_conf[pos].view(-1, 1)
        max_id = max_id[pos].view(-1, 1)
        inds = scores > thresh
        if len(inds) == 0:
            return np.empty((0,6))
        boxes = boxes[inds.view(-1, 1).expand(len(inds), 4)].view(-1, 4)
        scores = scores[inds].view(-1, 1)
        max_id = max_id[inds].view(-1, 1)
        c_dets = torch.cat((boxes, scores, max_id.float()), 1).cpu().numpy()
        img_classes = np.unique(c_dets[:, -1])
        output = None
        flag = False
        for cls in img_classes:
            cls_mask = np.where(c_dets[:, -1] == cls)[0]
            image_pred_class = c_dets[cls_mask, :]
            keep = nms(image_pred_class, 0.45, force_cpu=True)
            keep = keep[:50]
            image_pred_class = image_pred_class[keep, :]
            if not flag:
                output = image_pred_class
                flag = True
            else:
                output = np.concatenate((output, image_pred_class), axis=0)
        output[:, 0:2][output[:, 0:2]<0] = 0
        output[:, 2:4][output[:, 2:4]>1] = 1        
        scale = np.array([w, h, w, h])
        output[:, :4] = output[:, :4] * scale 
        t3 = time.time()
        print("transform_t:", round(t1-t0, 3), "detect_time:", round(t2-t1, 3), "nms_time:", round(t3-t2, 3))
    return output

def main():
    global args
    args = arg_parse()
    bgr_means = (104, 117, 123)
    dataset_name = args.dataset
    size = args.size
    top_k = args.top_k
    thresh = args.confidence_threshold
    use_refine = False
    if args.version.split("_")[0] == "refine":
        use_refine = True
    if dataset_name[0] == "V":
        cfg = cfg_dict["VOC"][args.version][str(size)]
        trainvalDataset = VOCDetection
        dataroot = VOCroot
        targetTransform = AnnotationTransform()
        valSet = datasets_dict["VOC2007"]
        classes = VOC_CLASSES
    else:
        cfg = cfg_dict["COCO"][args.version][str(size)]
        trainvalDataset = COCODetection
        dataroot = COCOroot
        targetTransform = None
        valSet = datasets_dict["COCOval"]
        classes = COCO_CLASSES
    num_classes = cfg['num_classes']
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    net = model_builder(args.version, cfg, "test", int(size), num_classes, args.channel_size)
    state_dict = torch.load(args.weights)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    detector = Detect(num_classes, 0, cfg, use_arm=use_refine)
    img_wh = cfg["img_wh"]
    ValTransform = BaseTransform(img_wh, bgr_means, (2, 0, 1))
    input_folder = args.images
    for item in os.listdir(input_folder)[:]: 
        img_path = os.path.join(input_folder, item)
        img = cv2.imread(img_path)
        dets = im_detect(img, net, detector, cfg, ValTransform, thresh)
        draw_img = draw_rects(img, dets, classes)
        out_img_name = "output_" + item
        save_path = os.path.join(save_folder, out_img_name)
        cv2.imwrite(save_path, img)   

if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)

