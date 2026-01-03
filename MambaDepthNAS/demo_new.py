import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os, sys
import torch

from torchvision import transforms

from networks.model import MambaDepth
from utils import (
    post_process_depth, flip_lr,
    convert_arg_line_to_args, str2list, unwrap_model
)


def parse_args():
    parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # input / output
    parser.add_argument('--image_path', type=str, required=True, help='image path / dir / txt list')
    parser.add_argument('--outdir', type=str, default='./files/vis_depth')
    parser.add_argument('--pred_only', dest='pred_only', action='store_true', help='only save prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='grayscale (no palette)')

    # device
    parser.add_argument('--devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')

    # model basic
    parser.add_argument('--model_name', type=str, default='MambaDepth')
    parser.add_argument('--encoder', type=str, default='SuperNet', help='SuperNet / VSSD_final / tiny07 ...')
    parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'kitti'])
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--max_depth', type=float, default=10.0)

    # input size (keep same semantics as your original MambaDepth demo)
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--input_width', type=int, default=640)

    # arch config (for SuperNet / VSSD_final)
    parser.add_argument('--mlp_ratio', type=str2list, default=[4.0, 4.0, 4.0, 4.0])
    parser.add_argument('--d_state', type=str2list, default=[64, 64, 64, -1])
    parser.add_argument('--ssd_expand', type=str2list, default=[4, 4, 4, -1])
    parser.add_argument('--depth', type=str2list, default=None)
    parser.add_argument('--width_multiplier', type=float, default=1.0)

    # post-process
    parser.add_argument('--post_process', dest='post_process', action='store_true', help='flip post-process')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    return args


def list_filenames(img_path: str):
    if os.path.isfile(img_path):
        if img_path.endswith('.txt'):
            with open(img_path, 'r') as f:
                filenames = [line.strip() for line in f.read().splitlines() if line.strip()]
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)

    # 过滤掉明显不是图片的文件
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    filenames = [p for p in filenames if p.lower().endswith(exts)]
    return filenames


def build_model(args, device):
    # VSSD_final 需要 final_config
    final_config = None
    if args.encoder == 'VSSD_final':
        depth_list = [
            (sum(x) if isinstance(x, (list, tuple)) else x)
            for x in args.depth
        ]
        final_config = {
            'mlp_ratio': args.mlp_ratio,
            'd_state': args.d_state + [-1],
            'ssd_expand': args.ssd_expand + [-1],
            'depth': depth_list,
        }

    model = MambaDepth(args, version=args.encoder, max_depth=args.max_depth, selected_config=final_config).to(device)

    # load ckpt
    if args.ckpt_path:
        ckpt = torch.load(open(args.ckpt_path, "rb"), map_location="cpu")
        key = 'model' if 'model' in ckpt else None
        sd = ckpt[key] if key else ckpt
        # 去掉 DDP 的 module.
        import re
        sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}
        incompatible = model.load_state_dict(sd, strict=True)
        print(f"[CKPT] Loaded: {args.ckpt_path}")
        print(f"[CKPT] missing/unexpected: {incompatible}")

    # SuperNet 需要 set_sample_config
    model_module = unwrap_model(model)
    if args.encoder == 'SuperNet':
        sample_config = {
            'mlp_ratio': args.mlp_ratio,
            'd_state': args.d_state,
            'expand': args.ssd_expand,
            'depth': args.depth,
        }
        if hasattr(model_module, "backbone") and hasattr(model_module.backbone, "set_sample_config"):
            model_module.backbone.set_sample_config(sample_config=sample_config)

    model.eval()
    return model


@torch.no_grad()
def infer_one(model, bgr_image, args, device):
    """
    输入: bgr_image uint8 HxWx3
    输出: depth float32 HxW （与输入 resize 后的尺寸一致）
    """
    if args.dataset == "kitti":
        h, w = bgr_image.shape[:2]
        top_margin = int(h - 352)
        left_margin = int((w - 1216) / 2)
        bgr_image = bgr_image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

    # BGR -> RGB
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # to tensor: 1x3xHxW, ImageNet normalize
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(x)
    x = x.unsqueeze(0).to(device)

    pred = model(x)

    if args.post_process:
        x_flip = flip_lr(x)
        pred_flip = model(x_flip)
        pred = post_process_depth(pred, pred_flip)

    depth = pred.squeeze().detach().float().cpu().numpy()  # HxW
    return depth


def depth_to_vis(depth, grayscale: bool, cmap):
    # min-max -> 0..255（每张图自适应显示）
    dmin, dmax = float(depth.min()), float(depth.max())
    denom = (dmax - dmin) + 1e-6
    depth_u8 = ((depth - dmin) / denom * 255.0).astype(np.uint8)

    if grayscale:
        vis = np.repeat(depth_u8[..., None], 3, axis=-1)  # HxWx3
    else:
        # cmap 输出 RGBA(0..1)，取 RGB，转 0..255，再 RGB->BGR 以便 cv2.imwrite
        vis = (cmap(depth_u8)[:, :, :3] * 255.0).astype(np.uint8)[:, :, ::-1]
    return vis


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[Device] {device}")

    os.makedirs(args.outdir, exist_ok=True)
    filenames = list_filenames(args.image_path)
    if len(filenames) == 0:
        raise RuntimeError(f"No images found from: {args.image_path}")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    model = build_model(args, device)

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if raw_image is None:
            print(f"[Skip] cannot read: {filename}")
            continue

        # 推理
        depth = infer_one(model, raw_image, args, device)

        depth_vis = depth_to_vis(depth, args.grayscale, cmap)

        out_name = os.path.splitext(os.path.basename(filename))[0] + '.png'
        out_path = os.path.join(args.outdir, out_name)

        if args.pred_only:
            cv2.imwrite(out_path, depth_vis)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([raw_image, split_region, depth_vis])
            cv2.imwrite(out_path, combined)

    print(f"[Done] saved to {args.outdir}")


if __name__ == '__main__':
    main()
