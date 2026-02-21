# -*- coding: utf-8 -*-
import argparse
import copy
import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

from networks.model import MambaDetection
from functools import partial

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError as e:
    raise ImportError("Please upgrade mmengine >= 0.6.0 to use get_model_complexity_info") from e


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True,
                        help="json list of backbone.selected_config")
    parser.add_argument("--base_config", type=str, required=True,
                        help="mmdet config .py")
    parser.add_argument("--num-images", type=int, default=50,
                        help="number of val images to average FLOPs/latency")
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup iterations for latency per image")
    parser.add_argument("--repeat", type=int, default=20,
                        help="repeat iterations for latency per image")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="override config keys, e.g. val_dataloader.dataset.data_root=...")
    parser.add_argument("--data-path", type=str, default=None,
                        help="optional dataset root path override")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()



@torch.no_grad()
def measure_latency(model, inputs, data_samples, warmup=10, repeat=20):
    model.eval()
    if inputs.is_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(inputs, data_samples=data_samples, mode="predict")
    if inputs.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = model(inputs, data_samples=data_samples, mode="predict")
    if inputs.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeat * 1000.0


@torch.no_grad()
def mean_flops_and_latency_on_val(
    model,
    data_loader,
    num_images=50,
    device="cuda",
    warmup=10,
    repeat=20,
    calc_latency=True,
):
    model.eval()

    flops_list = []
    latency_list = []
    params_str = None
    counted = 0

    for idx, data_batch in enumerate(data_loader):
        if counted >= num_images:
            break

        data = model.data_preprocessor(data_batch)

        # ---- FLOPs: mimic "new mmdet code" exactly ----
        orig_forward = model.forward
        try:
            model.forward = partial(model.forward, data_samples=data["data_samples"])
            out = get_model_complexity_info(
                model,
                input_shape=(3, 1280, 800),
                show_table=False,
                show_arch=False,
            )
        finally:
            model.forward = orig_forward

        flops_val = out.get("flops", None)
        if flops_val is None:
            flops_val = out.get("flops_num", None)
        if flops_val is None:
            raise RuntimeError(f"Unexpected keys from get_model_complexity_info: {list(out.keys())}")

        flops_list.append(float(flops_val))

        if params_str is None:
            params_str = out.get("params_str", None) or str(out.get("params", "N/A"))

        # --- Latency (optional) ---
        if calc_latency:
            inputs = data["inputs"]
            if device.startswith("cuda") and torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
            data_samples = data["data_samples"]
            lat_ms = measure_latency(model, inputs, data_samples, warmup, repeat)
            latency_list.append(lat_ms)

        counted += 1

    mean_flops = float(np.mean(flops_list)) if flops_list else float("nan")
    mean_latency = float(np.mean(latency_list)) if latency_list else float("nan")
    return mean_flops, mean_latency, params_str, counted

def build_val_dataloader(cfg: Config):
    # Ensure batch_size=1 for stable per-image accounting like your reference code
    if hasattr(cfg, "val_dataloader"):
        cfg.val_dataloader.batch_size = 1
    else:
        raise ValueError("Config has no val_dataloader")

    # Workdir required by Runner internals in some configs
    cfg.work_dir = tempfile.TemporaryDirectory().name
    return Runner.build_dataloader(cfg.val_dataloader)


@torch.no_grad()
def main():
    args = parse_args()
    logger = MMLogger.get_instance(name="MMLogger")

    # init mmdet registry scope
    init_default_scope("mmdet")
    register_all_modules(init_default_scope=True)

    # load list of searched configs
    with open(args.config_file) as f:
        config_list = json.load(f)

    # load base config
    cfg = Config.fromfile(args.base_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.data_path is not None:
        # basic override pattern (adjust if your dataset structure differs)
        cfg.merge_from_dict(dict(
            train_dataloader=dict(dataset=dict(data_root=args.data_path)),
            val_dataloader=dict(dataset=dict(data_root=args.data_path)),
        ))

    # build ONE dataloader (shared across model variants)
    val_loader = build_val_dataloader(cfg)

    results = []
    for one_cfg in config_list:
        model_cfg = copy.deepcopy(cfg.model)
        model_cfg.backbone.version = "VSSD_final"
        model_cfg.backbone.selected_config = one_cfg

        model = MODELS.build(model_cfg)

        if args.device.startswith("cuda") and torch.cuda.is_available():
            model = model.cuda()
        model = revert_sync_batchnorm(model)
        model.eval()

        mean_flops, mean_latency, params_str, used = mean_flops_and_latency_on_val(
            model,
            val_loader,
            num_images=args.num_images,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            calc_latency=True,
        )

        # Convert FLOPs to GFLOPs for readability (depends on mmengine definition)
        # NOTE: Some toolkits count MACs; keep numeric + string for transparency.
        results.append({
            "num_images": used,
            "mean_flops": mean_flops,          # raw number from mmengine analyzer
            "mean_flops_G": mean_flops / 1e9,  # convenience
            "mean_latency_ms": mean_latency,
            "params": params_str,
        })

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
