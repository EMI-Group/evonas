#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMSeg FP32 Latency & Throughput Benchmark
==========================================
Benchmark any MMSegmentation model (via official config + optional checkpoint)
for FP32 forward-pass latency (bs=1) and throughput (bs=N).

Usage example:
    python bench_mmseg_fp32.py \
        --config configs/upernet/upernet_r50_512x512_160k_ade20k.py \
        --checkpoint /path/to/ckpt.pth \
        --height 512 --width 512 \
        --throughput-bs 8 \
        --warmup-iters 20 --iters 100 \
        --cudnn-benchmark 1
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(arr, q):
    """Compute q-th percentile of a 1-D array (compatible with older numpy)."""
    return float(np.percentile(arr, q))


def get_gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A (CPU)"
    idx = device.index if device.index is not None else 0
    return torch.cuda.get_device_name(idx)


def build_model(config_path: str, checkpoint_path: str | None, device: torch.device):
    """
    Build mmseg model from config (+ optional checkpoint).
    Tries init_model first; falls back to Config + MODELS.build.
    """
    from mmengine import Config

    # Register all mmseg modules so that the registry is populated
    try:
        from mmseg.utils import register_all_modules
        register_all_modules(init_default_scope=True)
    except ImportError:
        pass

    # Attempt 1: mmseg.apis.init_model (official high-level API)
    try:
        from mmseg.apis import init_model
        model = init_model(config_path, checkpoint_path, device=str(device))
        return model
    except Exception:
        pass

    # Attempt 2: manual build
    cfg = Config.fromfile(config_path)
    from mmengine.registry import MODELS
    model = MODELS.build(cfg.model)
    if checkpoint_path is not None:
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(device)
    return model


def make_data_samples(batch_size: int, height: int, width: int):
    """Construct minimal SegDataSample list for mmseg forward."""
    from mmseg.structures import SegDataSample
    meta = {
        "img_shape": (height, width),
        "ori_shape": (height, width),
        "pad_shape": (height, width),
        "scale_factor": (1.0, 1.0),
        "flip": False,
        "flip_direction": None,
    }
    return [SegDataSample(metainfo=meta) for _ in range(batch_size)]


def forward_fn(model, inputs, data_samples):
    """
    Run model forward with fallback logic for different mmseg API versions.
    Priority:
      1) model(inputs, data_samples=data_samples, mode='predict')   # mmseg >=1.x
      2) model.forward(inputs)                                      # raw forward
    """
    try:
        return model(inputs, data_samples=data_samples, mode="predict")
    except TypeError:
        return model.forward(inputs)


class WrappedModel(torch.nn.Module):
    """Thin wrapper so ptflops can call model(x) without data_samples."""
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        try:
            return self.model(x, data_samples=self.data_samples, mode="predict")
        except TypeError:
            return self.model.forward(x)


def measure_complexity(model, channels, height, width):
    """
    Measure model params (M) and MACs/FLOPs (G) using ptflops.
    Returns (params_M, macs_G) or (None, None) if ptflops is unavailable.
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("[WARN] ptflops not installed, skipping Params/FLOPs measurement.")
        print("       Install with: pip install ptflops")
        return None, None

    input_shape = (channels, height, width)
    data_samples = make_data_samples(1, height, width)
    wrapped = WrappedModel(model, data_samples)
    macs, params = get_model_complexity_info(
        wrapped, input_shape,
        as_strings=False,
        backend="pytorch",
        print_per_layer_stat=False,
    )
    return params / 1e6, macs / 1e9  # Params(M), MACs(G)


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_latency(model, device, height, width, channels, batch_size,
                      warmup_iters, iters):
    """Per-iteration latency measurement (ms)."""
    inputs = torch.randn(batch_size, channels, height, width,
                         device=device, dtype=torch.float32)
    data_samples = make_data_samples(batch_size, height, width)

    # Warmup
    for _ in range(warmup_iters):
        forward_fn(model, inputs, data_samples)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn(model, inputs, data_samples)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(times)
    return {
        "mean": float(arr.mean()),
        "p50": percentile(arr, 50),
        "p90": percentile(arr, 90),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
        "raw": arr,
    }


@torch.inference_mode()
def benchmark_throughput(model, device, height, width, channels, batch_size,
                         warmup_iters, iters):
    """Per-iteration throughput measurement (img/s)."""
    inputs = torch.randn(batch_size, channels, height, width,
                         device=device, dtype=torch.float32)
    data_samples = make_data_samples(batch_size, height, width)

    # Warmup
    for _ in range(warmup_iters):
        forward_fn(model, inputs, data_samples)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn(model, inputs, data_samples)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(times)
    mean_ms = float(arr.mean())
    throughput = batch_size / (mean_ms / 1000.0)  # img/s
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(arr, 50),
        "p90_ms": percentile(arr, 90),
        "p95_ms": percentile(arr, 95),
        "p99_ms": percentile(arr, 99),
        "throughput": throughput,
        "raw": arr,
    }


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(run_results: list[dict], key_prefix: str = ""):
    """Aggregate stats across --num-runs repetitions."""
    keys = [k for k in run_results[0] if k != "raw"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in run_results]
        agg[f"{key_prefix}{k}_mean"] = float(np.mean(vals))
        agg[f"{key_prefix}{k}_std"] = float(np.std(vals))
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MMSeg FP32 Latency & Throughput Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="mmseg config file path")
    p.add_argument("--checkpoint", default=None, help="checkpoint file (optional)")
    p.add_argument("--height", type=int, default=512, help="input height")
    p.add_argument("--width", type=int, default=512, help="input width")
    p.add_argument("--latency-bs", type=int, default=1, help="batch size for latency test")
    p.add_argument("--throughput-bs", type=int, default=8, help="batch size for throughput test")
    p.add_argument("--warmup-iters", type=int, default=20, help="warmup iterations")
    p.add_argument("--iters", type=int, default=100, help="measurement iterations")
    p.add_argument("--device", type=str, default="cuda:0", help="device")
    p.add_argument("--cudnn-benchmark", type=int, default=1, choices=[0, 1],
                   help="enable cudnn.benchmark (0=off, 1=on)")
    p.add_argument("--channels", type=int, default=3, help="input channels (RGB=3)")
    p.add_argument("--fix-seed", type=int, default=None, help="fix random seed (optional)")
    p.add_argument("--tf32", type=int, default=0, choices=[0, 1],
                   help="allow TF32 on Ampere+ (0=off for strict FP32, 1=on)")
    p.add_argument("--num-runs", type=int, default=1,
                   help="repeat entire benchmark N times and report mean±std")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Seed ───────────────────────────────────────────────────────────
    if args.fix_seed is not None:
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.fix_seed)

    # ── FP32 enforcement ──────────────────────────────────────────────
    torch.set_default_dtype(torch.float32)
    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # ── cuDNN ─────────────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device(args.device)

    # ── Build model ───────────────────────────────────────────────────
    model = build_model(args.config, args.checkpoint, device)
    model.float()
    model.eval()

    # ── Model complexity ──────────────────────────────────────────────
    params_M, macs_G = measure_complexity(
        model, args.channels, args.height, args.width
    )

    # ── Environment info ──────────────────────────────────────────────
    gpu_name = get_gpu_name(device)
    config_basename = os.path.basename(args.config)

    complexity_str = ""
    if params_M is not None:
        complexity_str += f"Params       : {params_M:.2f} M\n"
    if macs_G is not None:
        complexity_str += f"MACs (FLOPs) : {macs_G:.2f} G\n"

    header = (
        f"{'=' * 50}\n"
        f" MMSeg FP32 Benchmark\n"
        f"{'=' * 50}\n"
        f"Config       : {args.config}\n"
        f"Checkpoint   : {args.checkpoint or '(none)'}\n"
        f"Device       : {args.device} ({gpu_name})\n"
        f"PyTorch      : {torch.__version__}\n"
        f"CUDA         : {torch.version.cuda}\n"
        f"{complexity_str}"
        f"Input        : {args.latency_bs}x{args.channels}x{args.height}x{args.width} (latency)\n"
        f"               {args.throughput_bs}x{args.channels}x{args.height}x{args.width} (throughput)\n"
        f"cudnn.bench  : {torch.backends.cudnn.benchmark}\n"
        f"TF32         : {'ON' if args.tf32 else 'OFF'}\n"
        f"Seed         : {args.fix_seed if args.fix_seed is not None else 'random'}\n"
        f"Warmup       : {args.warmup_iters} iters\n"
        f"Measure      : {args.iters} iters\n"
        f"Num runs     : {args.num_runs}\n"
        f"{'-' * 50}"
    )
    print(header)

    # ── Run benchmarks ────────────────────────────────────────────────
    latency_runs = []
    throughput_runs = []

    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n>>> Run {run_idx + 1}/{args.num_runs}")

        # Latency
        lat = benchmark_latency(
            model, device,
            args.height, args.width, args.channels,
            batch_size=args.latency_bs,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )
        latency_runs.append(lat)
        print(
            f"\n[Latency  bs={args.latency_bs}]\n"
            f"  mean : {lat['mean']:8.2f} ms\n"
            f"  p50  : {lat['p50']:8.2f} ms\n"
            f"  p90  : {lat['p90']:8.2f} ms\n"
            f"  p95  : {lat['p95']:8.2f} ms\n"
            f"  p99  : {lat['p99']:8.2f} ms"
        )

        # Throughput
        thr = benchmark_throughput(
            model, device,
            args.height, args.width, args.channels,
            batch_size=args.throughput_bs,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )
        throughput_runs.append(thr)
        print(
            f"\n[Throughput  bs={args.throughput_bs}]\n"
            f"  mean iter : {thr['mean_ms']:8.2f} ms\n"
            f"  p50  iter : {thr['p50_ms']:8.2f} ms\n"
            f"  p90  iter : {thr['p90_ms']:8.2f} ms\n"
            f"  p95  iter : {thr['p95_ms']:8.2f} ms\n"
            f"  p99  iter : {thr['p99_ms']:8.2f} ms\n"
            f"  throughput: {thr['throughput']:8.2f} img/s"
        )

    # ── Multi-run summary ─────────────────────────────────────────────
    if args.num_runs > 1:
        lat_agg = aggregate_runs(latency_runs, "lat_")
        thr_agg = aggregate_runs(throughput_runs, "thr_")
        print(
            f"\n{'=' * 50}\n"
            f" Aggregated over {args.num_runs} runs\n"
            f"{'=' * 50}\n"
            f"Latency  mean : {lat_agg['lat_mean_mean']:.2f} ± {lat_agg['lat_mean_std']:.2f} ms\n"
            f"Latency  p95  : {lat_agg['lat_p95_mean']:.2f} ± {lat_agg['lat_p95_std']:.2f} ms\n"
            f"Throughput    : {thr_agg['thr_throughput_mean']:.2f} ± {thr_agg['thr_throughput_std']:.2f} img/s"
        )

    print(f"\n{'=' * 50}")


if __name__ == "__main__":
    main()
