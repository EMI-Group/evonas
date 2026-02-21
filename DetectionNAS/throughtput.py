import torch
import json
import argparse
import gc
import time
import numpy as np
from ptflops import get_model_complexity_info

from mmdet.registry import MODELS
from networks.model import MambaDetection
from mmengine.registry import init_default_scope
init_default_scope('mmdet')

from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine import Config
from mmdet.structures import DetDataSample
import copy

""" Dont change any during searching !"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--base_config", type=str, required=True)
    return parser.parse_args()


def meta_for(H: int, W: int):
    """Build metainfo dict for DetDataSample."""
    return dict(
        img_shape=(H, W, 3),
        ori_shape=(H, W, 3),
        pad_shape=(H, W, 3),
        # (w_scale, h_scale)
        scale_factor=np.array([1.0, 1.0], dtype=np.float32),
        flip=False,
        flip_direction=None,
        batch_input_shape=(H, W),
    )


class WrappedModel(torch.nn.Module):
    """
    For ptflops: wrap mmdet model forward to a single-input interface.

    IMPORTANT:
    ptflops will call forward(x) with dummy input of batch=1 typically.
    Therefore data_samples must be created dynamically to match x.shape[0].
    """
    def __init__(self, model, H: int, W: int):
        super().__init__()
        self.model = model
        self.H, self.W = H, W

    def forward(self, x):
        bs = x.shape[0]
        data_samples = [DetDataSample(metainfo=meta_for(self.H, self.W)) for _ in range(bs)]
        return self.model(x, data_samples=data_samples, mode='predict')


@torch.no_grad()
def test_throughput_macs(
    model,
    input_shape=(1, 3, 800, 1280),
    device='cuda',
    warmup=100,
    repeat=500,
    batch_size=8,
    use_amp=True,
):
    """
    Throughput test for mmdet model (images/sec), keeping model/config intact.
    - Throughput uses median(iter_time) with cuda synchronize.
    - MACs/Params computed by ptflops with a dynamic data_samples wrapper.
    """
    model.eval()

    # Throughput uses batch_size
    input_tensor = torch.randn(batch_size, *input_shape[1:]).to(device)
    _, _, H, W = input_tensor.shape
    data_samples = [DetDataSample(metainfo=meta_for(H, W)) for _ in range(batch_size)]

    # warmup
    if use_amp:
        with torch.cuda.amp.autocast():
            for _ in range(warmup):
                _ = model(input_tensor, data_samples=data_samples, mode='predict')
    else:
        for _ in range(warmup):
            _ = model(input_tensor, data_samples=data_samples, mode='predict')
    torch.cuda.synchronize()

    # measure with per-iteration timing
    timer = []
    if use_amp:
        with torch.cuda.amp.autocast():
            for _ in range(repeat):
                tic = time.time()
                _ = model(input_tensor, data_samples=data_samples, mode='predict')
                torch.cuda.synchronize()
                timer.append(time.time() - tic)
    else:
        for _ in range(repeat):
            tic = time.time()
            _ = model(input_tensor, data_samples=data_samples, mode='predict')
            torch.cuda.synchronize()
            timer.append(time.time() - tic)

    throughput = int(batch_size / np.median(timer))  # images per second

    # ptflops (MACs/Params) - ptflops dummy input is usually batch=1
    wrapped_net = WrappedModel(model, H=H, W=W)
    macs, params = get_model_complexity_info(
        wrapped_net,
        tuple(input_shape[1:]),
        as_strings=False,
        backend='pytorch',
        print_per_layer_stat=False
    )

    # robust handling if ptflops returns None
    macs_G = None if macs is None else (macs / 1e9)
    params_M = None if params is None else (params / 1e6)

    return throughput, macs_G, params_M


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config_file) as f:
        config_list = json.load(f)

    cfg = Config.fromfile(args.base_config)

    input_shape = (1, 3, 800, 1280)

    results = []
    for config in config_list:
        model_cfg = copy.deepcopy(cfg.model)
        model_cfg.backbone.version = 'VSSD_final'
        model_cfg.backbone.selected_config = config

        model = MODELS.build(model_cfg)
        model.cuda()
        model.eval()

        throughput, macs, params = test_throughput_macs(
            model,
            input_shape=input_shape,
            device='cuda',
        )

        results.append({
            "throughput": throughput,
            "macs": macs,
            "params": params
        })

        # clean
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(results))


# CUDA_VISIBLE_DEVICES=5 python DetectionNAS/throughtput.py \
#   --config_file ./tmp/tmpabcd_GPU0.json \
#   --base_config DetectionNAS/configs/evo_mamba/mask_rcnn_evomamba_1x_coco.py

if __name__ == "__main__":
    main()