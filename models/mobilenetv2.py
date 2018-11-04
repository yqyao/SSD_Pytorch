from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.model_helper import weights_init


def add_extras(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    if size == '300':
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)]
    else:
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    return layers


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6,
                 activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(
            inplanes * t,
            inplanes * t,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(
            inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNet2(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self,
                 scale=1.0,
                 input_size=224,
                 t=6,
                 in_channels=3,
                 size=300,
                 activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNet2, self).__init__()

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.size = size

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        # assert (input_size % 32 == 0)

        self.c = [
            _make_divisible(ch * self.scale, 8) for ch in self.num_of_channels
        ]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(
            in_channels,
            self.c[0],
            kernel_size=3,
            bias=False,
            stride=self.s[0],
            padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        # self.bottlenecks = self._make_bottlenecks()
        self.bottlenecks = nn.ModuleList(self._make_bottlenecks())

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(
            1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(
            self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)

        self.extras = nn.ModuleList(
            add_extras(str(self.size), self.last_conv_out_ch))
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(
            inplanes=inplanes,
            outplanes=outplanes,
            stride=stride,
            t=t,
            activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(
                inplanes=outplanes,
                outplanes=outplanes,
                stride=1,
                t=6,
                activation=self.activation_type)
            modules[name] = module
        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = list()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(
            inplanes=self.c[0],
            outplanes=self.c[1],
            n=self.n[1],
            stride=self.s[1],
            t=1,
            stage=0)
        modules.append(bottleneck1)

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(
                inplanes=self.c[i],
                outplanes=self.c[i + 1],
                n=self.n[i + 1],
                stride=self.s[i + 1],
                t=self.t,
                stage=i)
            modules += module

        return modules

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        sources = list()
        for i in range(6):
            x = self.bottlenecks[i](x)
        sources.append(x)
        for i in range(6, 13):
            x = self.bottlenecks[i](x)
        sources.append(x)
        for i in range(13, len(self.bottlenecks)):
            x = self.bottlenecks[i](x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


def SSDMobilenetv2(size, channel_size='48'):
    return MobileNet2(size=size)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model3 = MobileNet2(size=300)
    with torch.no_grad():
        model3.eval()
        x = torch.randn(16, 3, 300, 300)
        model3.cuda()
        model3(x.cuda())
        import time
        st = time.time()
        for i in range(100):
            model3(x.cuda())
        print(time.time() - st)
        # print(model3(x))
