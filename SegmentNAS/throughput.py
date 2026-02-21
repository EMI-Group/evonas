import torch
import json
import argparse
import gc
from ptflops import get_model_complexity_info
import time
import numpy as np
from mmengine.registry import MODELS
from networks.model import MambaBackbone
from mmengine import Config
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)
import copy

register_all_modules(init_default_scope=True)
""" Dont change any during searching !"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--base_config", type=str, required=True)
    return parser.parse_args()


class WrappedModel(torch.nn.Module):
    """
    For ptflops: wrap mmseg model forward to a single-input interface.
    Note: ptflops will call forward(x) only.
    """
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        return self.model(x, data_samples=self.data_samples, mode='predict')


@torch.no_grad()
def test_throughput_macs(
    model,
    input_shape=(1, 3, 512, 512),
    device='cuda',
    warmup=100,
    repeat=500,
    batch_size=8,
    use_amp=True,
):
    """
    Throughput test for mmseg model (images/sec), keeping model/config intact.

    - Uses batch_size=8 by default (same as your depth example).
    - Measures per-iter time, uses median(timer) for stability.
    - Uses AMP autocast by default (consistent with your reference).
    """
    model.eval()

    # build input + data_samples for the batch
    input_tensor = torch.randn(batch_size, *input_shape[1:]).to(device)
    _, _, H, W = input_tensor.shape

    meta = {
        'img_shape': (H, W),
        'ori_shape': (H, W),
        'pad_shape': (H, W),
        'scale_factor': (1.0, 1.0),
        'flip': False,
        'flip_direction': None,
    }
    data_samples = [SegDataSample(metainfo=meta) for _ in range(batch_size)]

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

    # ptflops (MACs/Params) - keep consistent with mmseg forward signature
    wrapped_net = WrappedModel(model, data_samples)
    macs, params = get_model_complexity_info(
        wrapped_net,
        tuple(input_shape[1:]),
        as_strings=False,
        backend='pytorch',
        print_per_layer_stat=False
    )
    macs_G = macs / 1e9
    params_M = params / 1e6

    return throughput, macs_G, params_M


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config_file) as f:
        config_list = json.load(f)

    cfg = Config.fromfile(args.base_config)

    # ade20k
    input_shape = (1, 3, 512, 2048)

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


# CUDA_VISIBLE_DEVICES=5 python SegmentNAS/throughput.py \
#   --config_file ./tmp/tmpabcd_GPU0.json \
#   --base_config SegmentNAS/configs/upernet/upernet_evom_4xb4-160k_ade20k-512x512_fpn.py

if __name__ == "__main__":
    main()