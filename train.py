import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC, COCO, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, VOC_CLASSES, preproc, model_builder, pretrained_model, datasets_dict, cfg_dict
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
import numpy as np
import time
import os 
import sys
import pickle


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Receptive Field Block Net Training')
    parser.add_argument('-v', '--version', default='ssd_vgg',
                        help='')
    parser.add_argument('-s', '--size', default='300',
                        help='300 or 512 input size.')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO dataset')
    parser.add_argument('-c', '--channel_size', default='48',
                        help='channel_size')
    parser.add_argument(
        '--basenet', default='./weights/convert_darknet53.pth', help='pretrained base model')
    parser.add_argument('--jaccard_threshold', default=0.5,
                        type=float, help='Min Jaccard index for matching')
    parser.add_argument('-b', '--batch_size', default=32,
                        type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=8,
                        type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--lr', '--learning-rate',
                        default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--ngpu', default=2, type=int, help='gpus')
    parser.add_argument('--warmup', default=True,
                        type=bool, help='use warmup or not')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0,
                        type=int, help='resume iter for retraining')
    parser.add_argument('-max','--max_epoch', default=250,
                        type=int, help='max epoch for retraining')
    parser.add_argument('--visdom', default=False, help='Use visdom to for loss visualization')
    parser.add_argument('--send_images_to_visdom', default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

    parser.add_argument('--log_iters', default=True,
                        type=bool, help='Print the loss at each iteration')
    parser.add_argument('--save_folder', default='./weights/',
                        help='Location to save checkpoint models')
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size, iteration):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch <= 2:
        if args.warmup:
            iteration += iteration * epoch
            lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 2) 
        else:
            lr = args.lr
    else:
        div = 0
        if epoch > step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch > step_epoch[idx] and epoch <= step_epoch[idx+1]:
                    div = idx 
                    break
        lr = args.lr * (gamma ** div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, use_refine=False):
    net.train()
    begin = time.time()
    epoch_size = len(train_loader)
    for iteration, (imgs, targets, _) in enumerate(train_loader):
        t0 = time.time()
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma, epoch_size, iteration)
        imgs = imgs.cuda()
        imgs.requires_grad_()
        with torch.no_grad():
            targets = [anno.cuda() for anno in targets]
        output = net(imgs)
        optimizer.zero_grad()
        if not use_refine:
            ssd_criterion = criterion[0]
            loss_l, loss_c = ssd_criterion(output, targets) 
            loss = loss_l + loss_c
        else:
            arm_criterion = criterion[0]
            odm_criterion = criterion[1]
            arm_loss_l, arm_loss_c = arm_criterion(output, targets)
            odm_loss_l, odm_loss_c = odm_criterion(output, targets, use_arm=True, filter_object=True)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % 10 == 0:
            if not use_refine:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + ' || L: %.4f C: %.4f||' % (loss_l.item(), loss_c.item()) + 
                'iteration time: %.4f sec. ||' % (t1 - t0) + 'LR: %.5f' % (lr))
            else:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + 
                    '|| arm_L: %.4f arm_C: %.4f||' % (arm_loss_l.item(),arm_loss_c.item()) + ' odm_L: %.4f odm_C: %.4f||' % (odm_loss_l.item(), odm_loss_c.item()) + 
                  ' loss: %.4f||' % (loss.item()) +
                'iteration time: %.4f sec. ||' % (t1 - t0) + 'LR: %.5f' % (lr))                

def save_checkpoint(net, epoch, size):
    file_name = os.path.join(args.save_folder, args.version + "epoch_{}_{}".format(str(epoch), str(size))+ '.pth')
    torch.save(net.state_dict(), file_name)

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
    st = time.time()
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
    save_folder = args.save_folder
    batch_size = args.batch_size
    bgr_means = (104, 117, 123)
    weight_decay = 0.0005
    p = 0.6
    gamma = 0.1
    momentum = 0.9
    dataset_name = args.dataset
    size = args.size
    channel_size = args.channel_size
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
        top_k = 200
    else:
        cfg = cfg_dict["COCO"][args.version][str(size)]
        trainvalDataset = COCODetection
        dataroot = COCOroot
        targetTransform = None
        valSet = datasets_dict["COCOval"]
        top_k = 300
    num_classes = cfg['num_classes']
    start_epoch = args.resume_epoch
    epoch_step = cfg["epoch_step"]
    end_epoch = cfg["end_epoch"]
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    net = model_builder(args.version, cfg, "train", int(size), num_classes, args.channel_size)
    print(net)

    if args.resume_net == None:
        net.load_weights(pretrained_model[args.version])
    else:
        state_dict = torch.load(args.resume_net)
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
        print('Loading resume network...')
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=momentum, weight_decay=weight_decay)
    criterion = list()
    if use_refine:
        detector = Detect(num_classes, 0, cfg, use_arm=use_refine)
        arm_criterion = RefineMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

        odm_criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, 0.01, args.cuda)
        criterion.append(arm_criterion)
        criterion.append(odm_criterion)
    else:
        detector = Detect(num_classes, 0, cfg, use_arm=use_refine)
        ssd_criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
        criterion.append(ssd_criterion)
    TrainTransform = preproc(cfg["img_wh"], bgr_means, p)
    ValTransform = BaseTransform(cfg["img_wh"], bgr_means, (2, 0, 1))

    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, targetTransform, dataset_name)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=detection_collate)


    for epoch in range(start_epoch+1, end_epoch+1):
        train_dataset = trainvalDataset(dataroot, datasets_dict[dataset_name], TrainTransform, targetTransform, dataset_name)
        epoch_size = len(train_dataset)
        train_loader = data.DataLoader(train_dataset,
                                        batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        collate_fn=detection_collate)
        train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, use_refine)
        if (epoch % 10 == 0) or (epoch % 5 == 0 and epoch >= 200):
            save_checkpoint(net, epoch, size)
        if (epoch >= 200 and epoch % 10 == 0):
            eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, top_k, thresh=thresh, batch_size=batch_size)
    eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, top_k, thresh=thresh, batch_size=batch_size)
    save_checkpoint(net, end_epoch, size)
if __name__ == '__main__':
    main()


