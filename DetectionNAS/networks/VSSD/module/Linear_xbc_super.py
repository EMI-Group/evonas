import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearxBCSuper(nn.Linear):
    def __init__(self, super_in_dim, super_d_inner, super_ngroups, super_d_state, super_nheads, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, 2*super_d_inner + 2*super_ngroups*super_d_state + super_nheads, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_d_inner = super_d_inner
        self.super_ngroups = super_ngroups
        self.super_d_state = super_d_state
        self.super_nheads = super_nheads
        self.super_out_dim = 2 * self.super_d_inner + 2 * self.super_ngroups * self.super_d_state + self.super_nheads

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_d_inner = None
        self.sample_ngroups = None
        self.sample_d_state = None
        self.sample_nheads = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_d_inner, sample_d_state, sample_nheads):
        self.sample_in_dim = sample_in_dim
        self.sample_d_inner = sample_d_inner
        self.sample_ngroups = self.super_ngroups  # no change
        self.sample_d_state = sample_d_state
        self.sample_nheads = sample_nheads
        # update out_dim
        self.sample_out_dim = 2*self.sample_d_inner + 2*self.sample_ngroups*self.sample_d_state + self.sample_nheads

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = self.sample_in_proj_weight(self.weight)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = self.sample_in_proj_bias(self.bias)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops


    def sample_in_proj_weight(self, weight):
        """
        [z, xBC, dt] = [self.d_inner, self.d_inner + 2 * self.ngroups * self.sample_d_state, self.nheads]
        """
        sample_weight = weight[:, :self.sample_in_dim]
        # 第一段: z
        z = sample_weight[0 : self.sample_d_inner]

        # 第二段: xBC
        # xBC_start = self.super_d_inner
        # xBC = sample_weight[xBC_start : xBC_start + sample_d_inner + 2*sample_ngroups*sample_d_state]
        x_start = self.super_d_inner
        x = sample_weight[x_start : x_start + self.sample_d_inner]
        b_start = 2*self.super_d_inner
        b = sample_weight[b_start : b_start + self.sample_ngroups * self.sample_d_state]
        c_start = 2*self.super_d_inner + self.super_ngroups * self.super_d_state
        c = sample_weight[c_start : c_start + self.sample_ngroups * self.sample_d_state]

        # 第三段: dt
        dt_start = 2*self.super_d_inner + 2*self.super_ngroups*self.super_d_state 
        dt = sample_weight[dt_start : dt_start + self.sample_nheads]

        # sample_weight = torch.cat([z, xBC, dt], dim=0)
        sample_weight = torch.cat([z, x, b, c, dt], dim=0)
        return sample_weight


    def sample_in_proj_bias(self, bias):
        """
        [z, xBC, dt] = [self.d_inner, self.d_inner + 2 * self.ngroups * self.sample_d_state, self.nheads]
        """
        # 第一段: z
        z = bias[0 : self.sample_d_inner]

        # 第二段: xBC
        # xBC_start = self.super_d_inner
        # xBC = bias[xBC_start : xBC_start + sample_d_inner + 2*sample_ngroups*sample_d_state]
        x_start = self.super_d_inner
        x = bias[x_start : x_start + self.sample_d_inner]
        b_start = 2*self.super_d_inner
        b = bias[b_start : b_start + self.sample_ngroups * self.sample_d_state]
        c_start = 2*self.super_d_inner + self.super_ngroups * self.super_d_state
        c = bias[c_start : c_start + self.sample_ngroups * self.sample_d_state]

        # 第三段: dt
        dt_start = 2*self.super_d_inner + 2*self.super_ngroups*self.super_d_state 
        dt = bias[dt_start : dt_start + self.sample_nheads]

        # return torch.cat([z, xBC, dt], dim=0)
        return torch.cat([z, x, b, c, dt], dim=0)

# def sample_weight(weight, sample_in_dim, sample_out_dim):
#     sample_weight = weight[:, :sample_in_dim]
#     sample_weight = sample_weight[:sample_out_dim, :]

#     return sample_weight