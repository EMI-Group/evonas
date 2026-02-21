import torch
import json
import argparse
import gc
from networks.model import MambaDepth 
from ptflops import get_model_complexity_info
import time
''' Dont change any during searching !'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()

@torch.no_grad()
def test_latency_mac(model, input_shape=(1,3,224,224), device='cuda', warmup=30, repeat=50):
    model.eval()
    input_shape = (1, *input_shape[1:])
    input_tensor = torch.randn(*input_shape).to(device)

    # warmup
    for _ in range(warmup):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    # measure
    tic1 = time.perf_counter()
    for _ in range(repeat):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    tic2 = time.perf_counter()

    total_time = (tic2 - tic1)
    avg_latency = total_time / repeat * 1000  # ms
    # fps = repeat / total_time

    macs, params = get_model_complexity_info(model, tuple(input_shape[1:]), as_strings=False, backend='pytorch', print_per_layer_stat=False)

    macs_G = macs / 1e9
    params_M = params / 1e6

    return avg_latency, macs_G, params_M


@torch.no_grad()
def main():
    args = parse_args()

    # print("Raw string:", args.config_file)
    with open(args.config_file) as f:
        config_list = json.load(f)
    # print("Parsed dict:", config_list)

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

# CUDA_VISIBLE_DEVICES=5 python MambaDepthNAS/sub_test_latency.py --config_file ./tmp/tmpabcd_GPU0.json --dataset nyu
if __name__ == "__main__":
    main()
