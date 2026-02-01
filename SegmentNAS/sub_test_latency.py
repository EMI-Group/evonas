import torch
import json
import argparse
import gc
from ptflops import get_model_complexity_info
import time
from mmengine.registry import MODELS
from networks.model import MambaBackbone
from mmengine import Config
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)
import copy
''' Dont change any during searching !'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--base_config", type=str, required=True)
    return parser.parse_args()


class WrappedModel(torch.nn.Module):
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        return self.model(x, data_samples=self.data_samples, mode='predict')
    

@torch.no_grad()
def test_latency_mac(model, input_shape=(1,3,224,224), device='cuda', warmup=30, repeat=50):
    model.eval()
    input_shape = (1, *input_shape[1:])
    input_tensor = torch.randn(*input_shape).to(device)
    N, C, H, W = input_tensor.shape
    meta = {
        'img_shape': (H, W),
        'ori_shape': (H, W),
        'pad_shape': (H, W),
        'scale_factor': (1.0, 1.0,),
        'flip': False,
        'flip_direction': None,
    }
    data_samples = [SegDataSample(metainfo=meta) for _ in range(N)]

    # warmup
    for _ in range(warmup):
        _ = model(input_tensor, data_samples=data_samples, mode='predict')
    torch.cuda.synchronize()

    # measure
    tic1 = time.perf_counter()
    for _ in range(repeat):
        _ = model(input_tensor, data_samples=data_samples, mode='predict')
    torch.cuda.synchronize()
    tic2 = time.perf_counter()

    total_time = (tic2 - tic1)
    avg_latency = total_time / repeat * 1000  # ms
    # fps = repeat / total_time

    wrapped_net = WrappedModel(model, data_samples)
    macs, params = get_model_complexity_info(wrapped_net, tuple(input_shape[1:]), as_strings=False, backend='pytorch', print_per_layer_stat=False)

    macs_G = macs / 1e9
    params_M = params / 1e6

    return avg_latency, macs_G, params_M


@torch.no_grad()
def main():
    args = parse_args()

    # print("Raw string:", args.config_file)
    with open(args.config_file) as f:
        config_list = json.load(f)
    cfg = Config.fromfile(args.base_config)
    # print("Parsed dict:", config_list)

    input_shape = (1,3,512,512)  # ade20k

    results = []
    for config in config_list:
        model_cfg = copy.deepcopy(cfg.model)
        model_cfg.backbone.version = 'VSSD_final'
        model_cfg.backbone.selected_config = config

        model = MODELS.build(model_cfg)
        model.cuda()
        model.eval()

        latency, macs, params = test_latency_mac(model, input_shape, device='cuda')
        results.append({
            "latency": latency,
            "macs": macs,
            "params": params
        })
        # clean
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(results))

# CUDA_VISIBLE_DEVICES=5 python SegmentNAS/sub_test_latency.py --config_file ./tmp/tmpabcd_GPU0.json --base_config SegmentNAS/configs/upernet/upernet_evom_4xb4-160k_ade20k-512x512_fpn.py

if __name__ == "__main__":
    main()
