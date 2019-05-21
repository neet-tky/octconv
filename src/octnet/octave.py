import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from octconv import *

class SampleOct(nn.Module):
    def __init__(self, ch_in, ch_out, alpha=.5):
        super(SampleOct, self).__init__()

        self.features = nn.Sequential(
            OctConv2d(ch_in, 64, 3, stride=1, alpha=(.0, alpha)),
            ReLU_OctConv2d(64, alpha),
            MaxPool2d_OctConv2d(64, 2, 2, alpha=alpha),

            OctConv2d(64, 128, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(128, alpha),
            MaxPool2d_OctConv2d(128, 2, 2, alpha=alpha),

            OctConv2d(128, 256, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(256, alpha),
            OctConv2d(256, 256, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(256, alpha),
            MaxPool2d_OctConv2d(256, 2, 2, alpha=alpha),

            OctConv2d(256, 512, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(512, alpha),
            OctConv2d(512, 512, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(512, alpha),
            MaxPool2d_OctConv2d(512, 2, 2, alpha=alpha),

            OctConv2d(512, 512, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(512, alpha),
            OctConv2d(512, 512, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(512, alpha),
            MaxPool2d_OctConv2d(512, 2, 2, alpha=alpha),

            OctConv2d(512, 512, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(512, alpha),
            OctConv2d(512, 512, 3, stride=1, alpha=(alpha, alpha)),
            ReLU_OctConv2d(512, alpha),
            MaxPool2d_OctConv2d(512, 2, 2, alpha=alpha),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, ch_out)
        )

    def forward(self, x):
        fh, fl = self.features(x)

        x = torch.cat([F.max_pool2d(fh, 2, 2), fl], 1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, alpha=.5):
    return OctConv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, alpha=(alpha, alpha))

def conv1x1(in_planes, out_planes, stride=1, alpha=.5):
    return OctConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, alpha=(alpha, alpha))

class Bottleneck(nn.Module):
    expension = 4

    def __init__(self, inplanes, planes, alpha=.5, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = BN2d_OctConv2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width, alpha=alpha)
        self.bn1 = norm_layer(width, alpha=alpha)
        self.conv2 = conv3x3(width, width, stride=stride, alpha=alpha)
        self.bn2 = norm_layer(width, alpha=alpha)
        self.conv3 = conv3x3(width, planes * self.expension, alpha=alpha)
        self.bn3 = norm_layer(planes * self.expension, alpha=alpha)
        self.relu = ReLU_OctConv2d(planes * self.expension, alpha=alpha)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        xh, xl = x
        #print(xh.shape, xl.shape)
        y = self.conv1(x)
        #print(y[0].shape, y[1].shape)
        y = self.relu(self.bn1(y))
        y = self.relu(self.bn2(self.conv2(y)))
        yh, yl = self.bn3(self.conv3(y))

        if self.downsample is not None:
            xh, xl = self.downsample(x)

        yh += xh
        yl += xl

        y = (yh, yl)
        y = self.relu(y)
        return y

class OctResnet(nn.Module):
    def __init__(self, block, layers, alpha=.5, num_class=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):

        super(OctResnet, self).__init__()

        if norm_layer is None:
            norm_layer = BN2d_OctConv2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None"
                             "or a 3-element tuple, got{}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = OctConv2d(3, self.inplanes, kernel_size=2, stride=2, alpha=(.0, alpha))
        self.bn1 = norm_layer(self.inplanes, alpha)
        self.relu = ReLU_OctConv2d(self.inplanes, alpha)
        self.maxpool = MaxPool2d_OctConv2d(self.inplanes, 2, 2, alpha=alpha)

        self.layer1 = self._make_layer(block, 64, layers[0], alpha=alpha)
        self.layer2 = self._make_layer(block, 128, layers[1], alpha=alpha, stride=2, dilation=replace_stride_with_dilation)
        self.layer3 = self._make_layer(block, 256, layers[2], alpha=alpha, stride=2, dilation=replace_stride_with_dilation)
        self.layer4 = self._make_layer(block, 512, layers[3], alpha=alpha, stride=2, dilation=replace_stride_with_dilation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expension, num_class)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=False, alpha=.5):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilation:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expension:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expension, stride=stride, alpha=alpha),
                norm_layer(planes * block.expension, alpha=alpha),
            )

        layers = []
        layers.append(block(self.inplanes, planes, alpha=alpha, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))

        self.inplanes = planes * block.expension

        for _ in (1, blocks):
            layers.append(block(self.inplanes, planes, alpha=alpha, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        xh, xl = x
        x = torch.cat([F.max_pool2d(xh, 2, 2), xl], 1)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _octresnet(arch, inplanes, planes, pretrained, alpha=.5, **kwrgs):
    model = OctResnet(inplanes, planes, alpha, **kwrgs)

    if pretrained is not None:
        pass

    return model

def octres50(pretrained=None, alpha=.5, **kwargs):
    return _octresnet('octresnet50', Bottleneck, [3, 4, 6, 3], alpha=alpha, pretrained=pretrained, **kwargs)

if __name__=='__main__':
#    cnn = SampleOct(ch_in=3, ch_out=10, alpha=.75)
    cnn = octres50(alpha=.25)

    inputs = torch.randn(128, 3, 32, 32)
    pred_y = cnn(inputs)

    print(pred_y.size())