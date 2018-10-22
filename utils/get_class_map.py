import numpy as np
import os
import argparse
import os.path as osp


def check_size(submit_file):
    max_size = 60 * 1024 * 1024
    if osp.getsize(submit_file) > max_size:
        raise (
            IOError,
            "File size exceeds the specified maximum size, which is 60M for the server."
        )


def parse_submission(submit_file):
    with open(submit_file, 'r') as f:
        lines = f.readlines()
    submit_dict = dict()
    final_dict = dict()
    splitlines = [x.strip().split(' ') for x in lines]
    for idx, val in enumerate(splitlines):
        cls = str(int(float(val[1])))
        if cls not in submit_dict:
            submit_dict[cls] = list()
            final_dict[cls] = dict()
        submit_dict[cls].append(
            [val[0], val[2], val[3], val[4], val[5], val[6]])
    for k, v in submit_dict.items():
        image_ids = [x[0] for x in v]
        confidence = np.array([float(x[1]) for x in v])
        BB = np.array([[float(z) for z in x[2:]] for x in v])
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        final_dict[k]["image_ids"] = image_ids
        final_dict[k]["BB"] = np.array(BB)
    return final_dict


def parse_gt_annotation(gt_file):
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    info = [x.strip().split() for x in lines]
    gt = {}
    for item in info:
        img_id = item[0]
        obj_struct = {}
        obj_struct['class'] = item[1]
        obj_struct['bbox'] = [
            int(item[2]),
            int(item[3]),
            int(item[4]),
            int(item[5])
        ]
        if img_id not in gt:
            gt[img_id] = list()
        gt[img_id].append(obj_struct)
    return gt


def get_class_recs(recs, classname):
    npos = 0
    class_recs = {}
    for key in recs.keys():
        R = [obj for obj in recs[key] if obj['class'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[key] = {'bbox': bbox, 'det': det}
    return class_recs, npos


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval(submit_file, gt_file, ovthresh, classname):
    recs = parse_gt_annotation(gt_file)
    submit_result = parse_submission(submit_file)
    # get one class result
    class_recs, npos = get_class_recs(recs, classname)
    image_ids = submit_result[classname]["image_ids"]
    BB = submit_result[classname]["BB"]
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if image_ids[d] not in recs.keys():
            raise KeyError(
                "Can not find image {} in the groundtruth file, did you submit the result file for the right dataset?"
                .format(image_ids[d]))
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(rec, prec)
    return ap


def result_eval(submit_file, gt, class_list):
    ove_aap = []
    for ove in np.arange(0.5, 1.0, 0.05):
        cls_aap = []
        for cls in class_list:
            ap = eval(submit_file, gt, ove, cls)
            cls_aap.append(ap)
        cls_mAP = np.average(cls_aap)
        print("thresh", round(ove, 3), "map", round(cls_mAP * 100, 3))
        ove_aap.append(cls_mAP)
    mAP = np.average(ove_aap) * 100
    return round(mAP, 3)


if __name__ == '__main__':
    '''
    submit_file: image_id, class, score, xmin, ymin, xmax, ymax
    gt_file: image_id, class, xmin, ymin, xmax, ymax
    '''
    class_list = []
    for i in range(1, 61):
        class_list.append(str(i))
    submit_file = "./results/fpn_dcn_result.csv"
    gt_file = "./results/val_label.txt"
    check_size(submit_file)
    mAP = result_eval(submit_file, gt_file, class_list)
    out = {'Average AP': str(round(mAP, 3))}
    print(out)