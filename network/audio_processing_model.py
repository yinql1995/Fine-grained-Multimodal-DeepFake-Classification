import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
import scipy.io as sio
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import random

class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, mask=False):
        super(CONV, self).__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x, mask=False):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)
        device = x.device
        band_pass_filter = self.band_pass.to(device)

        # Frequency masking: We randomly mask (1/5)th of no. of sinc filters channels (70)
        if (mask == True):
            for i1 in range(1):
                A = np.random.uniform(0, 14)
                A = int(A)
                A0 = random.randint(0, band_pass_filter.shape[0] - A)
                band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        # print('filter', self.filters.size())

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
            self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                                   out_channels=nb_filts[1],
                                   kernel_size=(2, 3),
                                   padding=(1, 1),
                                   stride=1)
        self.selu = nn.SELU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=nb_filts[1],
                                kernel_size=(2, 3),
                                padding=(1, 1),
                                stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],

                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
            out = self.conv1(out)
        else:
            x = x
            out = self.conv_1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class audio_model(nn.Module):
    def __init__(self, num_nodes=4):
        super(audio_model, self).__init__()

        '''
        Sinc conv. layer
        '''
        self.conv_time = CONV(
                              out_channels=50,
                              kernel_size=128,
                              in_channels=1
                              )

        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.selu = nn.SELU(inplace=True)

        # Note that here you can also use only one encoder to reduce the network parameters which is jsut half of the 0.44M (mentioned in the paper). I was doing some subband analysis and forget to remove the use of two encoders.  I also checked with one encoder and found same results.

        self.encoder1 = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=[32, 32], first=True)),
            nn.Sequential(Residual_block(nb_filts=[32, 32])),
            nn.Sequential(Residual_block(nb_filts=[32, 64])),

            nn.Sequential(Residual_block(nb_filts=[64, 64])),
            nn.Sequential(Residual_block(nb_filts=[64, 64])),
            nn.Sequential(Residual_block(nb_filts=[64, 32]))
        )

        self.num_nodes=num_nodes
    def forward(self, x, Freq_aug=False):
        """
        x= (#bs,samples)
        """

        # follow sincNet recipe
        # print('1', x.shape)

        nb_samp = x.shape[0]
        len_seq = x.shape[1]

        x = x.view(nb_samp, 1, len_seq)

        # Freq masking during training only

        if (Freq_aug == True):
            x = self.conv_time(x, mask=True)  # (#bs,sinc_filt(70),64472)
        else:
            x = self.conv_time(x, mask=False)
        """
        Different with the our RawNet2 model, we interpret the output of sinc-convolution layer as 2-dimensional image with one channel (like 2-D representation).
        """
        # print(x.size())

        x = x.unsqueeze(dim=1)  # 2-D (#bs,1,sinc-filt(70),64472)
        # print(x.size())
        x = F.max_pool2d(torch.abs(x), (3, 3))  # [#bs, C(1),F(23),T(21490)]

        x = self.first_bn(x)
        x = self.selu(x)
        # print(x.size())

        temp = torch.chunk(x, self.num_nodes, dim=3)
        out = []

        for i in range(self.num_nodes):
            # encoder structure for spectral GAT
            t = self.encoder1(temp[i])
            # print(t.size())

            # max-pooling along time with absolute value  (Attention in spectral part)
            x_max, _ = torch.max(torch.abs(t), dim=3)  # [#bs, C(64), F(23)]
            out.append(x_max)
        out = torch.stack(out, dim=1)
        # print(out.size())
        out = out.view(out.size(0),out.size(1), -1)
        # print(out.size())
        return out

    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []
        # def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts,
                                         first=first))
            if i == 0: nb_filts[0] = nb_filts[1]

        return nn.Sequential(*layers)

if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = audio_model(num_nodes=4)
    model(torch.randn(2, 64000))

