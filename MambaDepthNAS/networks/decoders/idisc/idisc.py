"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_decoder import BasePixelDecoder
from .id_module import AFP, ISD


class IDisc(nn.Module):
    def __init__(
        self,
        afp: nn.Module,
        pixel_decoder: nn.Module,
        isd: nn.Module,
        afp_min_resolution=1,
        eps: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.afp = afp
        self.pixel_decoder = pixel_decoder
        self.isd = isd
        self.afp_min_resolution = afp_min_resolution

    def invert_encoder_output_order(
        self, xs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(
        self, decoder_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[self.afp_min_resolution :])

    def forward(
        self,
        encoder_outputs,
        original_shape,
    ):
        encoder_outputs = self.invert_encoder_output_order(encoder_outputs)  # TODO need?

        # DefAttn Decoder + filter useful resolutions (usually skip the lowest one)
        fpn_outputs, decoder_outputs = self.pixel_decoder(encoder_outputs)

        decoder_outputs = self.filter_decoder_relevant_resolutions(decoder_outputs)
        fpn_outputs = self.filter_decoder_relevant_resolutions(fpn_outputs)

        idrs = self.afp(decoder_outputs)
        outs = self.isd(fpn_outputs, idrs)

        out_lst = []
        for out in outs:
            if out.shape[1] == 1:
                out = F.interpolate(
                    torch.exp(out),
                    size=outs[-1].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                out = self.normalize_normals(
                    F.interpolate(
                        out,
                        size=outs[-1].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            out_lst.append(out)

        out = F.interpolate(
            torch.mean(torch.stack(out_lst, dim=0), dim=0),
            original_shape,
            # Legacy code for reproducibility for normals...
            mode="bilinear" if out.shape[1] == 1 else "bicubic",
            align_corners=True,
        )

        return out

    def normalize_normals(self, norms):
        min_kappa = 0.01
        norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
        norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
        kappa = F.elu(kappa) + 1.0 + min_kappa
        norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
        return norms


    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):

        pixel_encoder_embed_dims = config["model"]["pixel_encoder"]["embed_dims"]
        pixel_decoder = BasePixelDecoder.build(config)

        afp = AFP.build(config)
        isd = ISD.build(config)

        return deepcopy(
            cls(
                pixel_decoder=pixel_decoder,
                afp=afp,
                isd=isd,
                afp_min_resolution=len(pixel_encoder_embed_dims)
                - config["model"]["isd"]["num_resolutions"],
            )
        )
