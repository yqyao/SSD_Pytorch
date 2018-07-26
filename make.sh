#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace
# if you use anaconda3 maybe you need add this
# change code like https://github.com/rbgirshick/py-faster-rcnn/issues/706
mv nms/cpu_nms.cpython-36m-x86_64-linux-gnu.so nms/cpu_nms.so
mv nms/gpu_nms.cpython-36m-x86_64-linux-gnu.so nms/gpu_nms.so
cd ..
