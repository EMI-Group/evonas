import json, random, argparse, os
from typing import List, Optional, Dict, Any

def has_valid_ann_for_maskrcnn(ann: Dict[str, Any]) -> bool:
    # 只把非 crowd 的注释作为“有效性判断”依据（crowd 留在输出里作为忽略区域）
    if ann.get('iscrowd', 0) == 1:
        return False
    # bbox 是否正常
    bbox = ann.get('bbox', [])
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False
    # segmentation 是否存在（Mask R-CNN 训练需要）
    seg = ann.get('segmentation', None)
    if seg is None:
        return False
    if isinstance(seg, list):
        return len(seg) > 0
    if isinstance(seg, dict):  # RLE
        return bool(seg.get('counts'))
    return False

def build_id2anns(annotations: List[Dict[str, Any]]):
    d = {}
    for a in annotations:
        d.setdefault(a['image_id'], []).append(a)
    return d

def main(src: str, dst: str, n: int, seed: int, need_mask: bool, ids: Optional[List[int]]):
    with open(src, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    anns   = coco['annotations']
    cats   = coco['categories']
    id2anns = build_id2anns(anns)

    # 候选图片：至少有一个“有效标注”
    def img_is_candidate(img_id: int) -> bool:
        if img_id not in id2anns:
            return False
        if not need_mask:
            # 只要有任意标注就算候选
            return len(id2anns[img_id]) > 0
        return any(has_valid_ann_for_maskrcnn(a) for a in id2anns[img_id])

    candidates = [img_id for img_id in images.keys() if img_is_candidate(img_id)]
    if not candidates:
        raise RuntimeError("未找到满足条件的图片，请检查标注文件。")

    # 选择图片 id
    if ids and len(ids) > 0:
        sel = []
        miss = []
        allow = set(candidates)
        for x in ids:
            if x in allow:
                sel.append(x)
            else:
                miss.append(x)
        if miss:
            print(f"[WARN] 以下指定 id 不在候选中或无有效标注，将被忽略：{miss}")
        if len(sel) == 0:
            raise RuntimeError("指定的 ids 里没有可用的。")
    else:
        if len(candidates) < n:
            raise RuntimeError(f"候选图片不足 {n} 张，当前只有 {len(candidates)}。")
        random.seed(seed)
        sel = random.sample(candidates, n)

    sel_set = set(sel)
    sub_images = [images[i] for i in sel]
    sub_anns   = [a for a in anns if a['image_id'] in sel_set]

    out = {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'images': sub_images,
        'annotations': sub_anns,
        'categories': cats  # 不改类别映射，最稳
    }

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[OK] 写入 {dst} | images={len(sub_images)}, anns={len(sub_anns)}")

# python make_coco_subset.py \
#   --src ./instances_val2017.json \
#   --dst ./instances_val2000.json \
#   -n 2000 --seed 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, required=True, help='原始 COCO 标注，如 instances_val2017.json')
    ap.add_argument('--dst', type=str, required=True, help='输出子集标注，如 instances_val8.json')
    ap.add_argument('-n', '--num-images', type=int, default=8, help='抽取图片数量')
    ap.add_argument('--seed', type=int, default=0, help='随机种子')
    ap.add_argument('--no-mask', action='store_true', help='不强制要求有 mask（仅做 bbox 也可用）')
    ap.add_argument('--ids', type=str, default='', help='指定图片 id 列表，逗号分隔，如 "397133,37777,252219"')
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(',')] if args.ids.strip() else None
    main(args.src, args.dst, args.num_images, args.seed, need_mask=not args.no_mask, ids=ids)
