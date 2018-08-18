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
from data import VOCroot, COCOroot, VOC, COCO, AnnotationTransform, COCOAnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, VOC_CLASSES, preproc, model_builder, pretrained_model, datasets_dict, cfg_dict
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
import numpy as np
from utils.timer import Timer
import time
import os 
import sys


def arg_parse():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--weights', default='weights/ssd_darknet_300.pth',
                        type=str, help='Trained state_dict file path to open')

    parser.add_argument('-v', '--version', default='ssd_vgg',
                        help='dense_ssd or origin_ssd version.')
    parser.add_argument('-s', '--size', default='300',
                        help='300 or 512 input size.')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC ,VOC0712++ or COCO dataset')
    parser.add_argument('-b', '--batch_size', default=32,
                        type=int, help='Batch size for training')
    parser.add_argument('-c', '--channel_size', default='48',
                        help='channel_size 32_1, 32_2, 48, 64, 96, 128')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='File path to save results')
    parser.add_argument('--num_workers', default=8,
                        type=int, help='Number of workers used in dataloading')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', default=200, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, help='Use cuda to train model')
    parser.add_argument('--retest', default=False, type=bool,
                        help='test cache results')
    args = parser.parse_args()
    return args

def eval_net(val_dataset, val_loader, net, detector, cfg, transform, max_per_image=300, thresh=0.01, batch_size=1):
    net.eval()
    num_images = len(val_dataset)
    num_classes = cfg['num_classes']
    eval_save_folder = "./eval/"
    if not os.path.exists(eval_save_folder):
        os.mkdir(eval_save_folder)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    det_file = os.path.join(eval_save_folder, 'detections.pkl')

    _t = {'im_detect': Timer(), 'misc': Timer()}

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        val_dataset.evaluate_detections(all_boxes, eval_save_folder)
        return

    for idx, (imgs, _, img_info) in enumerate(val_loader):
        with torch.no_grad():
            t1 = time.time()
            x = imgs
            x = x.cuda()
            output = net(x)
            t4 = time.time()
            boxes, scores = detector.forward(output)
            t2 = time.time()
            for k in range(boxes.size(0)):
                i = idx * batch_size + k
                boxes_ = boxes[k]
                scores_ = scores[k]
                boxes_ = boxes_.cpu().numpy()
                scores_ = scores_.cpu().numpy()
                img_wh = img_info[k]
                scale = np.array([img_wh[0], img_wh[1], img_wh[0], img_wh[1]])
                boxes_ *= scale
                for j in range(1, num_classes):
                    inds = np.where(scores_[:, j] > thresh)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes_[inds]
                    c_scores = scores_[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False)
                    keep = nms(c_dets, 0.45, force_cpu=True)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets
            t3 = time.time()
            detect_time = t2 - t1
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'
                    .format(i + 1, num_images, forward_time, detect_time, nms_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)
    print("detect time: ", time.time() - st)

def main():
    global args
    args = arg_parse()
    bgr_means = (104, 117, 123)
    dataset_name = args.dataset
    size = args.size
    use_refine = False
    batch_size = args.batch_size
    num_workers = args.num_workers
    if args.version.split("_")[0] == "refine":
        use_refine = True
    if dataset_name[0] == "V":
        cfg = cfg_dict["VOC"][args.version][str(size)]
        trainvalDataset = VOCDetection
        dataroot = VOCroot
        targetTransform = AnnotationTransform()
        valSet = datasets_dict["VOC2007"]
    else:
        cfg = cfg_dict["COCO"][args.version][str(size)]
        trainvalDataset = COCODetection
        dataroot = COCOroot
        targetTransform = None
        valSet = datasets_dict["COCOval"]
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
    ValTransform = BaseTransform(cfg["img_wh"], bgr_means, (2, 0, 1))
    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, targetTransform, "val")
    val_loader = data.DataLoader(val_dataset,
                                 batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=detection_collate)
    top_k = 300
    thresh = 0.01
    eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, top_k, thresh=thresh, batch_size=batch_size)

if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)

