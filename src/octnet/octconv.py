# implementation of OctConv
# reference:
# https://githuh.com/motokimura/oct_convpytorch/blob/master/octconv.py


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OctConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, bias=True, alpha=(.5, .5)):
        super(OctConv2d, self).__init__()

        #alpha
        self.alpha_in, self.alpha_out = alpha
        self.ch_in_xl = int(self.alpha_in * ch_in)
        self.ch_in_xh = ch_in - self.ch_in_xl

        self.ch_out_yl = int(self.alpha_out * ch_out)
        self.ch_out_yh = ch_out - self.ch_out_yl

        # padding
        padding = (kernel_size - stride) // 2
        self.padding = padding
        self.kernel_size = kernel_size
        self.stridef = stride
        self.wh2h, self.wh2l, self.wl2l, self.wl2h = None, None, None, None
        if not (self.ch_in_xh == 0 or self.ch_out_yh == 0):
            self.wh2h = nn.Conv2d(self.ch_in_xh, self.ch_out_yh, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        if not (self.ch_in_xh == 0 or self.ch_out_yl == 0):
            self.wh2l = nn.Conv2d(self.ch_in_xh, self.ch_out_yl, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        if not (self.ch_in_xl == 0 or self.ch_out_yl == 0):
            self.wl2l = nn.Conv2d(self.ch_in_xl, self.ch_out_yl, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        if not (self.ch_in_xl == 0 or self.ch_out_yh == 0):
           self.wl2h = nn.Conv2d(self.ch_in_xl, self.ch_out_yh, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        xh, xl = None, None

        # input features maps for changing alpha
        if self.ch_in_xl == 0:
            xh = x

        elif self.ch_in_xh == 0:
            xl = x

        else:
            xh, xl = x

        # apply conv
        yh2h = yh2l = yl2l = yl2h = 0
        if self.wh2h is not None:
            yh2h = self.wh2h(xh)
            #print('yh2h' + str(yh2h.shape))

        if self.wh2l is not None:
            yh2l = self.wh2l(F.avg_pool2d(xh, 2))
            #print('yh2l' + str(yh2l.shape))

        if self.wl2l is not None:
            yl2l = self.wl2l(xl)
            #print('yl2l' + str(yl2l.shape))

        if self.wl2h is not None:
            yl2h = F.interpolate(self.wl2h(xl), scale_factor=2, mode='nearest')
            #print('yl2h' + str(yl2h.shape))


        yh = yh2h + yl2h
        yl = yl2l + yh2l

        #print(yh.shape, yl.shape)
        # output FMs
        if self.ch_out_yl == 0:
            return yh

        elif self.ch_out_yh == 0:
            return yl

        else:
            return (yh, yl)


# need function
class MaxPool2d_OctConv2d(nn.Module):
    def __init__(self, channels, kernel_size, stride, alpha=.5):
        super(MaxPool2d_OctConv2d, self).__init__()

        assert 0 <= alpha <= 1, 'Alpha must be in iterval [0, 1]'
        self.ch_l = int(alpha * channels)
        self.ch_h = channels - self.ch_l

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if self.ch_h == 0 or self.ch_l == 0:
            return F.max_pool2d(x, self.kernel_size, self.stride)

        xh, xl = x
        #print(xh.shape, xl.shape)
        yh = F.max_pool2d(xh, self.kernel_size, self.stride)
        yl = F.max_pool2d(xl, self.kernel_size, self.stride)

        return (yh, yl)

class AvgPool2d_OctConv2d(nn.Module):
    def __init__(self, channels, kernel_size, stride, alpha=.5):
        super(AvgPool2d_OctConv2d, self).__init__()

        assert 0 <= alpha <= 1, 'Alpha must be in iterval [0, 1]'
        self.ch_l = int(alpha * channels)
        self.ch_h = channels - self.ch_l

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if self.ch_h == 0 or self.ch_l == 0:
            return F.avg_pool2d(x, self.kernel_size, self.stride)

        xh, xl = x
        yh = F.avg_pool2d(xh, self.kernel_size, self.stride)
        yl = F.avg_pool2d(xl, self.kernel_size, self.stride)

        return (yh, yl)

class Upsample_OctConv2d(nn.Module):
    def __init__(self, channels, scale_factor, mode='bilinear', alpha=.5):
        super(Upsample_OctConv2d, self).__init__()

        assert 0 <= alpha <= 1, 'Alpha must be in iterval [0, 1]'
        self.ch_l = int(alpha * channels)
        self.ch_h = channels - self.ch_l

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if self.ch_h == 0 or self.ch_l == 0:
            return F.interpolate(x, scale_factor=self.scale_facotr, mode=self.mode)

        xh, xl = x
        yh = F.interpolate(xh, scale_factor=self.scale_facotr, mode=self.mode)
        yl = F.interpolate(xl, scale_factor=self.scale_facotr, mode=self.mode)

        return (yh, yl)

class BN2d_OctConv2d(nn.Module):
    def __init__(self, channels, alpha=.5):
        super(BN2d_OctConv2d, self).__init__()

        assert 0 <= alpha <= 1, 'Alpha must be in iterval [0, 1]'
        self.ch_l = int(alpha * channels)
        self.ch_h = channels - self.ch_l

        self.bn2d_h = nn.BatchNorm2d(self.ch_h) if self.ch_h > 0 else None
        self.bn2d_l = nn.BatchNorm2d(self.ch_l) if self.ch_l > 0 else None

    def forward(self, x):
        if self.bn2d_h is None:
            return self.bn2d_l(x)

        if self.bn2d_l is None:
            return self.bn2d_h(x)

        xh, xl = x
        #print(xh.shape, xl.shape, self.ch_l, self.ch_h)

        yh = self.bn2d_h(xh)
        yl = self.bn2d_l(xl)

        return (yh, yl)

class ReLU_OctConv2d(nn.Module):
    def __init__(self, channels, alpha=.5):
        super(ReLU_OctConv2d, self).__init__()

        assert 0 <= alpha <= 1, 'Alpha must be in iterval [0, 1]'
        self.ch_l = int(alpha * channels)
        self.ch_h = channels - self.ch_l

    def forward(self, x):
        if self.ch_l == 0 or self.ch_h == 0:
            return F.relu(x, inplace=True)

        xh, xl = x
        yh = F.relu(xh, inplace=True)
        yl = F.relu(xl, inplace=True)

        return (yh, yl)

'''
class SampleOct(nn.Module):
    def __init__(self):
        super(SampleOct, self).__init__()

        self.oc_in = OctConv2d(3, 16, 3, stride=1, alpha=(.0, .5))

        self.oc1 = OctConv2d(16, 32, 3, stride=1, alpha=(.5, .75))
        self.relu1 = ReLU_OutConv2d(32, .75)
        self.pool1 = MaxPool2d_OctConv2d(32, 2, 2, .75)

        self.oc2 = OctConv2d(32, 64, 7, stride=1, alpha=(.75, .5))
        self.relu2 = ReLU_OutConv2d(64, .5)
        self.pool2 = MaxPool2d_OctConv2d(64, 2, 2, .5)

        self.oc_out = OctConv2d(64, 10, 3, stride=1, alpha=(.5, .5))

    def forward(self, x):
        h = self.oc_in(inputs)
        h = self.relu1(self.oc1(h))
        h = self.pool1(h)
        h = self.relu2(self.oc2(h))
        h = self.pool2(h)

        return self.oc_out(h)

if __name__=='__main__':
    octconv = SampleOct()

    inputs = torch.randn(16, 3, 224, 224)
    outputs = octconv(inputs)
    print(outputs[0].shape, outputs[1].shape)
'''