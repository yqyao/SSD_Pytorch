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
    parser.add_argument('-c', '--channel_size', default='48',
                        help='channel_size 32_1, 32_2, 48, 64, 96, 128')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='File path to save results')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', default=200, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, help='Use cuda to train model')
    parser.add_argument('--retest', default=False, type=bool,
                        help='test cache results')
    args = parser.parse_args()
    return args

def eval_net(val_dataset, net, detector, cfg, transform, max_per_image=300, thresh=0.01):
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

    for i in range(num_images):
        with torch.no_grad():
            t0 = time.time()
            img = val_dataset.pull_image(i)
            w, h = img.shape[1], img.shape[0]
            x = transform(img).unsqueeze(0)
            x = x.cuda()
            t1 = time.time()
            output = net(x)
            boxes, scores = detector.forward(output)
            detect_time = _t['im_detect'].toc()
            t2 = time.time()
            boxes = boxes[0]
            scores = scores[0]
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            # scale each detection back up to the image
            scale = np.array([w, h, w, h])
            boxes *= scale
            t3 = time.time()
            for j in range(1, num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
                keep = nms(c_dets, 0.45, force_cpu=True)
                keep = keep[:50]
                c_dets = c_dets[keep, :]
                all_boxes[j][i] = c_dets
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
            nms_time = _t['misc'].toc()
            t4 = time.time()
        read_time = round(t1 - t0, 3)
        detect_time = round(t2 - t1, 3)
        copy_time = round(t3 - t2, 3)
        nms_time = round(t4- t3, 3)
        all_time = round(t4 - t0, 3)

        if i % 20 == 0:
            info = "{}/{} | read_t: {}s |det_t: {}s | copy_t: {}s | nms_t: {}s | all_t: {}s".format(str(i+1), str(num_images), read_time, detect_time, copy_time, nms_time, all_time)
            print(info)
        if i == 0:
            begin = time.time()
    print("detect all time", time.time() - begin)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)


def main():
    global args
    args = arg_parse()
    bgr_means = (104, 117, 123)
    dataset_name = args.dataset
    size = args.size
    use_refine = False
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
    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, targetTransform, "test")
    top_k = 300
    thresh = 0.01
    eval_net(val_dataset, net, detector, cfg, ValTransform, top_k, thresh=thresh)

if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)

