import torch
from mmengine.structures import InstanceData
from mmdet.models.utils import unpack_gt_instances

@torch.no_grad()
def build_targets_from_rois(model_module, rois, batch_data_samples):
    """
    给每个 roi 找到它该学的 GT（分类标签+回归目标）
    rois: Tensor [N,5] (batch_id, x1,y1,x2,y2)  —— 来自 student 当前训练路径
    返回:
      labels: [N]  (0..C-1 for fg, -1 for bg/ignore)
      bbox_targets: [N,4]  delta/encode space
      pos_mask: [N] bool  正样本 roi
    """
    roi_head = model_module.roi_head
    assigner = roi_head.bbox_assigner
    coder = roi_head.bbox_head.bbox_coder

    batch_gt_instances, batch_gt_instances_ignore, _ = unpack_gt_instances(batch_data_samples)

    device = rois.device
    N = rois.size(0)
    labels = torch.full((N,), -1, device=device, dtype=torch.long)
    bbox_targets = torch.zeros((N, coder.encode_size), device=device, dtype=torch.float32)
    pos_mask = torch.zeros((N,), device=device, dtype=torch.bool)

    batch_ids = rois[:, 0].long()
    priors = rois[:, 1:5]  # xyxy

    num_imgs = len(batch_gt_instances)
    for i in range(num_imgs):
        inds = torch.nonzero(batch_ids == i, as_tuple=False).squeeze(1)
        if inds.numel() == 0:
            continue

        pri_i = priors[inds]
        gt_i = batch_gt_instances[i]
        ign_i = None if batch_gt_instances_ignore is None else batch_gt_instances_ignore[i]

        # assigner 需要一个 InstanceData，里边有 priors
        rpn_results = InstanceData()
        rpn_results.priors = pri_i
        rpn_results.scores = pri_i.new_ones((pri_i.size(0),))  # 占位即可

        assign_result = assigner.assign(rpn_results, gt_i, ign_i)
        gt_inds = assign_result.gt_inds  # 0:bg, >0 matched gt index

        pos = gt_inds > 0
        pos_mask[inds] = pos

        if pos.any():
            matched_gt = gt_i.bboxes[gt_inds[pos] - 1]
            bbox_targets[inds[pos]] = coder.encode(pri_i[pos], matched_gt)

            # 这里 labels 用于 class-specific reg gather
            labels[inds[pos]] = gt_i.labels[gt_inds[pos] - 1]

    return labels, bbox_targets, pos_mask


def kd_cls_soft_ce(cls_s, cls_t, bg_weight=1.5):
    '''让学生的分类分布接近 teacher 的分类分布'''
    # cls_s/cls_t: [N, C+1]，最后一列通常是 bg
    # 数值稳定：用 float32 计算 softmax/log_softmax
    cls_s = cls_s.float()
    cls_t = cls_t.float()
    p_t = torch.softmax(cls_t, dim=-1)
    log_p_s = torch.log_softmax(cls_s, dim=-1)

    w = torch.ones_like(p_t)
    bg_idx = p_t.size(1) - 1
    w[:, bg_idx] = bg_weight
    return -(w * p_t * log_p_s).sum(dim=-1).mean()


def _pick_reg(pred, labels, num_classes, reg_class_agnostic=False):
    '''把 class-specific 回归挑成每个 roi 对应的那 4 个数'''
    # pred: [N,4] or [N,4*C]
    if reg_class_agnostic or pred.size(1) == 4:
        return pred
    pred = pred.view(pred.size(0), num_classes, 4)
    idx = torch.arange(pred.size(0), device=pred.device)
    return pred[idx, labels.clamp(min=0)]


def kd_reg_teacher_bounded(reg_s, reg_t, bbox_targets, pos_mask, labels,
                           num_classes, reg_class_agnostic=False,
                           margin=0.0, nu=0.5):
    if pos_mask.sum() == 0:
        return reg_s.new_tensor(0.0)

    rs = _pick_reg(reg_s, labels, num_classes, reg_class_agnostic)[pos_mask].float()
    rt = _pick_reg(reg_t, labels, num_classes, reg_class_agnostic)[pos_mask].float()
    tgt = bbox_targets[pos_mask].float()

    err_s = ((rs - tgt) ** 2).sum(dim=1)
    err_t = ((rt - tgt) ** 2).sum(dim=1)
    gate = (err_s + margin > err_t).float()
    return nu * 0.5 * (err_s * gate).mean()
