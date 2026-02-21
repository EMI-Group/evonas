import torch
import json
import argparse
import gc
import time
import numpy as np
from networks.model import MambaDepth
from ptflops import get_model_complexity_info
''' Dont change any during searching !'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()

@torch.no_grad()
def test_throughput(model, input_shape=(1,3,224,224), device='cuda', warmup=100, repeat=500, batch_size=8):
    """
    Test model throughput (images/second).
    batch_size=8, AMP autocast, median-based timing (consistent with throughput.py).
    """
    model.eval()
    input_tensor = torch.randn(batch_size, *input_shape[1:]).to(device)

    # warmup
    with torch.cuda.amp.autocast():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # measure with per-iteration timing
    timer = []
    with torch.cuda.amp.autocast():
        for _ in range(repeat):
            tic = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            timer.append(time.time() - tic)

    throughput = int(batch_size / np.median(timer))  # images per second

    macs, params = get_model_complexity_info(model, tuple(input_shape[1:]), as_strings=False, backend='pytorch', print_per_layer_stat=False)

    macs_G = macs / 1e9
    params_M = params / 1e6

    return throughput, macs_G, params_M


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config_file) as f:
        config_list = json.load(f)

    if args.dataset == 'kitti':
        input_shape = (1,3,352,1216)
        args.input_height = 352
        args.input_width = 1216
    elif args.dataset == 'nyu':
        input_shape = (1,3,480,640)
        args.input_height = 480
        args.input_width = 640

    results = []
    for config in config_list:
        model = MambaDepth(args=args, version='VSSD_final', selected_config=config)
        model.cuda()
        model.eval()

        throughput, macs, params = test_throughput(model, input_shape, device='cuda')
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


if __name__ == "__main__":
    main()
