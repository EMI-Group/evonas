import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GConvSuper(nn.Conv2d):
    def __init__(self, super_in_channels, super_out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, scale=False,
                   super_d_inner=None, super_ngroups=None, super_d_state=None):
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        super().__init__(
            in_channels=super_in_channels,
            out_channels=super_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # no change params
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # change params
        self.super_in_channels = super_in_channels
        self.super_out_channels = super_out_channels
        self.super_d_inner = super_d_inner
        self.super_ngroups = super_ngroups
        self.super_d_state = super_d_state

        self.sample_in_channels = None
        self.sample_out_channels = None
        self.sample_d_inner = None
        self.sample_ngroups = None
        self.sample_d_state = None

        self.samples = {}
        self.scale = scale
    
    def set_sample_config(self, sample_in_channels, sample_d_inner, sample_ngroups, sample_d_state):
        self.sample_in_channels = sample_in_channels
        self.sample_out_channels = sample_in_channels  # in_C == out_C
        assert sample_in_channels == sample_d_inner + 2 * sample_ngroups * sample_d_state
        self.sample_d_inner = sample_d_inner
        self.sample_ngroups = sample_ngroups
        self.sample_d_state = sample_d_state
        self._sample_parameters()

    def _sample_parameters(self):
        '''[x, B, C] = [d_inner, ngroups * sample_d_state, ngroups * sample_d_state]'''
        # weight
        weight = self.weight[:, :self.sample_in_channels, :, :]

        x_part = weight[0 : self.sample_d_inner]

        b_start = self.super_d_inner
        B_part = weight[b_start : b_start + self.sample_ngroups*self.sample_d_state]

        c_start = self.super_d_inner + self.super_ngroups*self.super_d_state
        C_part = weight[c_start : c_start + self.sample_ngroups*self.sample_d_state]

        weight = torch.cat([x_part, B_part, C_part], dim=0)

        # bias
        if self.bias is not None:
            bias_x = self.bias[0 : self.sample_d_inner]
            bias_B = self.bias[b_start : b_start + self.sample_ngroups*self.sample_d_state]
            bias_C = self.bias[c_start : c_start + self.sample_ngroups*self.sample_d_state]

            bias = torch.cat([bias_x, bias_B, bias_C], dim=0)
        else:
            bias = None

        self.samples['weight'] = weight
        self.samples['bias'] = bias

        self.sample_scale = self.super_out_channels / self.sample_out_channels
        return self.samples

    def forward(self, x):
        self._sample_parameters()
        return F.conv2d(
            x,
            self.samples['weight'],
            self.samples['bias'],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.sample_in_channels  # note!
        ) * (self.sample_scale if self.scale else 1)
