# SSD Pytorch
A [PyTorch](http://pytorch.org/) implementation of SSDs (include original ssd, DRFNet, RefineDet)

<!-- <img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/> -->

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch-0.4.0](http://pytorch.org/)  by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below.
- Compile the nms and install coco tools:

```shell
cd SSD_Pytorch
# if you use anaconda3, maybe you need https://github.com/rbgirshick/py-faster-rcnn/issues/706
./make.sh
pip install pycocotools

```

Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:

```Shell
'nvcc': ['-arch=sm_52',

```

## Datasets
To make things easy, we provide a simple VOC dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

##### Merge VOC2007 and VOC2012

```Shell
move all images in VOC2007 and VOC2012 into VOCROOT/VOC0712/JPEGImages
move all annotations in VOC2007 and VOC2012 into VOCROOT/VOC0712/JPEGImages/Annotations
rename and merge some txt VOC2007 and VOC2012 ImageSets/Main/*.txt to VOCROOT/VOC0712/JPEGImages/ImageSets/Main/*.txt
the merged txt list as follows:
2012_test.txt, 2007_test.txt, 0712_trainval_test.txt, 2012_trainval.txt, 0712_trainval.txt

```
### COCO Dataset
Install the MS COCO dataset at /path/to/coco from [official website](http://mscoco.org/), default is ~/data/COCO. Following the [instructions](https://github.com/rbgirshick/py-faster-rcnn/blob/77b773655505599b94fd8f3f9928dbf1a9a776c7/data/README.md) to prepare *minival2014* and *valminusminival2014* annotations. All label files (.json) should be under the COCO/annotations/ folder. It should have this basic structure
```Shell
$COCO/
$COCO/cache/
$COCO/annotations/
$COCO/images/
$COCO/images/test2015/
$COCO/images/train2014/
$COCO/images/val2014/
```
*UPDATE*: The current COCO dataset has released new *train2017* and *val2017* sets which are just new splits of the same image sets. 


## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- ResNet pre-trained basenet weight file is available at [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth), [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth), [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)
- By default, we assume you have downloaded the file in the `SSD_Pytorch/weights/pretrained_models` dir:

```Shell
mkdir weights
cd weights
mkdir pretrained_models

wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
mv download_weights pretrained_models
```

- To train SSD_Pytorch using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py --cfg ./configs/ssd_vgg_voc.yaml
```

- Note:
  All training configs are in ssd_vgg_voc.yaml, you can change it by yourself.

- To evaluate a trained network:

```Shell
python eval.py --cfg ./configs/ssd_vgg_voc.yaml --weights ./eval_weights
```

- To detect one images

```
 # you need put some images in ./images
python demo.py --cfg ./configs/ssd_vgg_voc.yaml --images ./images --save_folder ./output

```
You can specify the parameters listed in the `eval.py` or `demo.py` file by flagging them or manually changing them.  

## Performance

#### VOC2007 Test

##### mAP

we retrained some models, so it's different from the origin paper
size = 300

|ssd_vgg|ssd_res|ssd_darknet|drf_ssd_vgg|drf_ssd_res|refine_drf_vgg|refine_ssd_vgg| 
|:-:      |:-:      |:-:          |:-:          |:-:          |:-:|:-:   |          
| 77.5%   | 77.0    | 77.6%       | 79.6 %      | 79.0%       |80.2% |80.4 %        |




## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo): 
  * [ssd.pytorch]((https://github.com/amdegroot/ssd.pytorch)),
    [RFBNet](https://github.com/ruinmessi/RFBNet)
    [Chainer](https://github.com/Hakuyume/chainer-ssd),
    [torchcv](https://github.com/kuangliu/torchcv)
  ) 






