import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import torch.fft as fft
from .attention import MultiheadPosAttention, WindowMultiheadPosAttention
from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn.bricks.transformer import FFN

'''
code from ScaleKD
'''
class FreqMaskingDistillLossv2(nn.Module):

    def __init__(self,
                 alpha,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.,
                 dis_freq='high',
                 num_heads=8
                 ):
        super(FreqMaskingDistillLossv2, self).__init__()
        self.alpha = alpha
        self.dis_freq = dis_freq
        self.self_query = self_query

        self.projector_0 = AttentionProjector(
            student_dims, teacher_dims, pos_dims,
            window_shapes=window_shapes, self_query=self_query,
            softmax_scale=softmax_scale[0], num_heads=num_heads
        )
        self.projector_1 = AttentionProjector(
            student_dims, teacher_dims, pos_dims,
            window_shapes=window_shapes, self_query=self_query,
            softmax_scale=softmax_scale[1], num_heads=num_heads
        )

    def forward(self,
                preds_S,
                preds_T,
                query_s=None,
                query_f=None,
                ):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*Hs*Ws, student's feature map
            preds_T(Tensor): Bs*C*Ht*Wt, teacher's feature map (Ht,Wt 动态)
        """

        # [改动点2] projector 需要知道 teacher 的 Ht,Wt，从 preds_T 实时取
        preds_S_spat = self.projector_0(preds_S, preds_T, query=query_s)
        preds_S_freq = self.projector_1(preds_S, preds_T, query=query_f)

        spat_loss = self.get_spat_loss(preds_S_spat, preds_T)
        freq_loss = self.get_freq_loss(preds_S_freq, preds_T)

        return spat_loss, freq_loss

    def get_spat_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        N = preds_S.shape[0]
        N, C, H, W = preds_T.shape


        preds_S = preds_S.permute(0,2,1).contiguous().view(*preds_T.shape)

        preds_S = F.normalize(preds_S, dim=1)
        preds_T = F.normalize(preds_T, dim=1)

        dis_loss_arch_st = loss_mse(preds_S, preds_T)/N
        dis_loss_arch_st = dis_loss_arch_st * self.alpha[0]

        return dis_loss_arch_st


    def get_freq_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device

        dct = DCT(resolution=(H, W), device=device)

        preds_S = preds_S.permute(0,2,1).contiguous().view(*preds_T.shape)

        preds_S_freq = dct.forward(preds_S)
        preds_T_freq = dct.forward(preds_T)

        preds_S_freq[:,:,0,0]=0
        preds_T_freq[:,:,0,0]=0

        preds_S = dct.inverse(preds_S_freq)
        preds_T = dct.inverse(preds_T_freq)

        preds_S = F.normalize(preds_S, dim=1, p=2)
        preds_T = F.normalize(preds_T, dim=1, p=2)

        dis_loss = loss_mse(preds_S, preds_T)/N

        dis_loss = dis_loss * self.alpha[1]

        return dis_loss
    
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class AttentionProjector(nn.Module):
    def __init__(self,
                 student_dims,
                 teacher_dims,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.,
                 num_heads=8,
                 base_hw=(48, 48),   # 可调：建议 >= 你常见最大Ht/Wt
                 ):
        super(AttentionProjector, self).__init__()

        self.student_dims = student_dims
        self.teacher_dims = teacher_dims

        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 1),
                                      nn.ReLU(),
                                      )

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU())

        self.pos_embed = PosOnlyQuery(pos_dims=student_dims, base_hw=(32,32))
        self.pos_attention = WindowMultiheadPosAttention(teacher_dims, num_heads=num_heads, input_dims=student_dims, pos_dims=pos_dims, window_shapes=window_shapes, softmax_scale=softmax_scale)
        self.ffn = FFN(embed_dims=teacher_dims, feedforward_channels=teacher_dims * 4)

        self.norm = nn.LayerNorm([teacher_dims])

        if self_query:
            self.query = PosOnlyQuery(pos_dims=pos_dims, base_hw=base_hw)
        else:
            self.query = None

    def forward(self, x, preds_T, query=None):
        """
        x: (B, student_dims, Hs, Ws)
        preds_T: (B, teacher_dims, Ht, Wt) 只用于取 (Ht,Wt)
        query: 可选，支持：
          - (B, Ht*Wt, pos_dims) 或 (B, pos_dims, Ht, Wt)
        return:
          - fea_S: (B, Ht*Wt, teacher_dims)
        """
        _, _, Ht, Wt = preds_T.shape
        B, _, Hs, Ws = x.shape

        pos_emb = self.query(B, Ht, Wt, x.device, x.dtype)  # (B, pos_dims, Ht, Wt)


        preds_S = self.proj_student(x) + self.pos_embed(B, Hs, Ws, x.device, x.dtype)
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)  # 前者需要学生尺度，后者需要教师尺度
        fea_S = self.ffn(self.norm(fea_S))

        return fea_S


class PosOnlyQuery(nn.Module):
    def __init__(self, pos_dims, base_hw=(32, 32)):
        super().__init__()
        Hb, Wb = base_hw
        self.grid = nn.Parameter(torch.zeros(1, pos_dims, Hb, Wb))
        trunc_normal_(self.grid, std=0.02)

    def forward(self, B, Ht, Wt, device, dtype):
        pos = F.interpolate(self.grid.to(device=device, dtype=dtype),
                            size=(Ht, Wt), mode="bilinear", align_corners=False)
        return pos.repeat(B, 1, 1, 1)  # (B, pos_dims, Ht, Wt)


class DCT():
    def __init__(self, resolution, device, norm=None, bias=False):
        self.resolution = resolution
        self.norm = norm
        self.device = device

        I_h = torch.eye(self.resolution[1], device=self.device)
        self.forward_transform_h = nn.Linear(resolution[1], resolution[1], bias=bias).to(self.device)
        self.forward_transform_h.weight.data = self._dct(I_h, norm=self.norm).data.t()
        self.forward_transform_h.weight.requires_grad = False

        I_w = torch.eye(self.resolution[0], device=self.device)
        self.forward_transform_w = nn.Linear(resolution[0], resolution[0], bias=bias).to(self.device)
        self.forward_transform_w.weight.data = self._dct(I_w, norm=self.norm).data.t()
        self.forward_transform_w.weight.requires_grad = False

        self.inverse_transform_h = nn.Linear(resolution[1], resolution[1], bias=bias).to(self.device)
        self.inverse_transform_h.weight.data = self._idct(I_h, norm=self.norm).data.t()
        self.inverse_transform_h.weight.requires_grad = False

        self.inverse_transform_w = nn.Linear(resolution[0], resolution[0], bias=bias).to(self.device)
        self.inverse_transform_w.weight.data = self._idct(I_w, norm=self.norm).data.t()
        self.inverse_transform_w.weight.requires_grad = False

    def _dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def _idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape)

    def forward(self, x):
        X1 = self.forward_transform_h(x)
        X2 = self.forward_transform_w(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)

    def inverse(self, x):
        X1 = self.inverse_transform_h(x)
        X2 = self.inverse_transform_w(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)