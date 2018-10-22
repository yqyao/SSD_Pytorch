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
from data import COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
import datetime
from models.model_builder import SSD
import yaml


def arg_parse():
    parser = argparse.ArgumentParser(description='SSD Training')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument('--ngpu', default=2, type=int, help='gpus')
    parser.add_argument(
        '--resume_net', default=None, help='resume net for retraining')
    parser.add_argument(
        '--resume_epoch',
        default=0,
        type=int,
        help='resume iter for retraining')

    parser.add_argument(
        '--save_folder',
        default='./weights/',
        help='Location to save checkpoint models')
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size,
                         iteration):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch <= cfg.TRAIN.WARMUP_EPOCH:
        if cfg.TRAIN.WARMUP:
            iteration += (epoch_size * (epoch - 1))
            lr = 1e-6 + (cfg.SOLVER.BASE_LR - 1e-6) * iteration / (
                epoch_size * cfg.TRAIN.WARMUP_EPOCH)
        else:
            lr = cfg.SOLVER.BASE_LR
    else:
        div = 0
        if epoch > step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch > step_epoch[idx] and epoch <= step_epoch[idx + 1]:
                    div = idx
                    break
        lr = cfg.SOLVER.BASE_LR * (gamma**div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma,
          end_epoch, cfg):
    net.train()
    begin = time.time()
    epoch_size = len(train_loader)
    for iteration, (imgs, targets, _) in enumerate(train_loader):
        t0 = time.time()
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma,
                                  epoch_size, iteration)
        imgs = imgs.cuda()
        imgs.requires_grad_()
        with torch.no_grad():
            targets = [anno.cuda() for anno in targets]
        output = net(imgs)
        optimizer.zero_grad()
        if not cfg.MODEL.REFINE:
            ssd_criterion = criterion[0]
            loss_l, loss_c = ssd_criterion(output, targets)
            loss = loss_l + loss_c
        else:
            arm_criterion = criterion[0]
            odm_criterion = criterion[1]
            arm_loss_l, arm_loss_c = arm_criterion(output, targets)
            odm_loss_l, odm_loss_c = odm_criterion(
                output, targets, use_arm=True, filter_object=True)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        iteration_time = t1 - t0
        all_time = ((end_epoch - epoch) * epoch_size +
                    (epoch_size - iteration)) * iteration_time
        eta = str(datetime.timedelta(seconds=int(all_time)))
        if iteration % 10 == 0:
            if not cfg.MODEL.REFINE:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' +
                      repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                      ' || L: %.4f C: %.4f||' %
                      (loss_l.item(), loss_c.item()) +
                      'iteration time: %.4f sec. ||' % (t1 - t0) +
                      'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))
            else:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' +
                      repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                      '|| arm_L: %.4f arm_C: %.4f||' %
                      (arm_loss_l.item(), arm_loss_c.item()) +
                      ' odm_L: %.4f odm_C: %.4f||' %
                      (odm_loss_l.item(), odm_loss_c.item()) +
                      ' loss: %.4f||' % (loss.item()) +
                      'iteration time: %.4f sec. ||' % (t1 - t0) +
                      'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))


def save_checkpoint(net, epoch, size, optimizer):
    save_name = os.path.join(
        args.save_folder,
        cfg.MODEL.TYPE + "_epoch_{}_{}".format(str(epoch), str(size)) + '.pth')
    torch.save({
        'epoch': epoch,
        'size': size,
        'batch_size': cfg.TRAIN.BATCH_SIZE,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_name)


def eval_net(val_dataset,
             val_loader,
             net,
             detector,
             cfg,
             transform,
             max_per_image=300,
             thresh=0.01,
             batch_size=1):
    net.eval()
    num_images = len(val_dataset)
    num_classes = cfg.MODEL.NUM_CLASSES
    eval_save_folder = "./eval/"
    if not os.path.exists(eval_save_folder):
        os.mkdir(eval_save_folder)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
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
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(
                                            np.float32, copy=False)
                    keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=True)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets
            t3 = time.time()
            detect_time = t2 - t1
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(
                    i + 1, num_images, forward_time, detect_time, nms_time))
    print("detect time: ", time.time() - st)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)



def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    save_folder = args.save_folder
    batch_size = cfg.TRAIN.BATCH_SIZE
    bgr_means = cfg.TRAIN.BGR_MEAN
    p = 0.6
    gamma = cfg.SOLVER.GAMMA
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    size = cfg.MODEL.SIZE
    thresh = cfg.TEST.CONFIDENCE_THRESH
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        top_k = 200
    else:
        trainvalDataset = COCODetection
        top_k = 300
    dataset_name = cfg.DATASETS.DATA_TYPE
    dataroot = cfg.DATASETS.DATAROOT
    trainSet = cfg.DATASETS.TRAIN_TYPE
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    start_epoch = args.resume_epoch
    epoch_step = cfg.SOLVER.EPOCH_STEPS
    end_epoch = cfg.SOLVER.END_EPOCH
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = SSD(cfg)
    print(net)
    if cfg.MODEL.SIZE == '300':
        size_cfg = cfg.SMALL
    else:
        size_cfg = cfg.BIG
    optimizer = optim.SGD(
        net.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=momentum,
        weight_decay=weight_decay)
    if args.resume_net != None:
        checkpoint = torch.load(args.resume_net)
        state_dict = checkpoint['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loading resume network...')
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()
    cudnn.benchmark = True

    criterion = list()
    if cfg.MODEL.REFINE:
        detector = Detect(cfg)
        arm_criterion = RefineMultiBoxLoss(cfg, 2)
        odm_criterion = RefineMultiBoxLoss(cfg, cfg.MODEL.NUM_CLASSES)
        criterion.append(arm_criterion)
        criterion.append(odm_criterion)
    else:
        detector = Detect(cfg)
        ssd_criterion = MultiBoxLoss(cfg)
        criterion.append(ssd_criterion)

    TrainTransform = preproc(size_cfg.IMG_WH, bgr_means, p)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))

    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, dataset_name)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_dataset = trainvalDataset(dataroot, trainSet, TrainTransform,
                                        dataset_name)
        epoch_size = len(train_dataset)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=detection_collate)
        train(train_loader, net, criterion, optimizer, epoch, epoch_step,
              gamma, end_epoch, cfg)
        if (epoch % 10 == 0) or (epoch % 5 == 0 and epoch >= 200):
            save_checkpoint(net, epoch, size, optimizer)
        if (epoch >= 50 and epoch % 10 == 0):
            eval_net(
                val_dataset,
                val_loader,
                net,
                detector,
                cfg,
                ValTransform,
                top_k,
                thresh=thresh,
                batch_size=batch_size)
    save_checkpoint(net, end_epoch, size, optimizer)


if __name__ == '__main__':
    main()
