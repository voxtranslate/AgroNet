import torch
import torch.nn as nn


# =============================================
# LOSS FUNCTIONS
# =============================================

# ═══════════════════════════════════════════════════════════════════════════
# 8. TAL LABEL ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

class TaskAlignedLabelAssigner(nn.Module):
    """YOLOv8 Task-Aligned Learning assignment."""
    def __init__(self, top_k: int = 10, alpha: float = 0.5, beta: float = 6.0,
                 eps: float = 1e-9):
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    @torch.no_grad()
    def forward(
        self,
        cls_preds:     torch.Tensor,
        box_preds:     torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_boxes:      torch.Tensor,
        gt_labels:     torch.Tensor,
        img_size:      int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, max_gt, _ = gt_boxes.shape
        A            = anchor_points.shape[0]
        nc           = cls_preds.shape[-1]
        device       = cls_preds.device

        anchor_xy     = anchor_points * stride_tensor
        target_boxes  = torch.zeros(B, A, 4,  device=device)
        target_scores = torch.zeros(B, A, nc, device=device)
        fg_mask       = torch.zeros(B, A,     device=device, dtype=torch.bool)
        target_gt_idx = torch.zeros(B, A,     device=device, dtype=torch.long)

        for b in range(B):
            gt_b  = gt_boxes[b];   lbl_b = gt_labels[b]
            valid = lbl_b >= 0;    n_gt  = valid.sum().item()
            if n_gt == 0:
                continue
            gt_b = gt_b[valid];    lbl_b = lbl_b[valid]

            ax  = anchor_xy[:, 0].unsqueeze(0)
            ay  = anchor_xy[:, 1].unsqueeze(0)
            in_box = ((ax > gt_b[:, 0:1]) & (ax < gt_b[:, 2:3]) &
                      (ay > gt_b[:, 1:2]) & (ay < gt_b[:, 3:4]))

            iou_mat = _pairwise_iou_xyxy(box_preds[b].unsqueeze(0),
                                          gt_b.unsqueeze(1))

            cls_gt = cls_preds[b, :, lbl_b].permute(1, 0)
            align  = ((cls_gt.clamp(0, 1) ** self.alpha) *
                      (iou_mat.clamp(0, 1) ** self.beta) *
                      in_box.float())

            topk_vals, _ = align.topk(min(self.top_k, A), dim=1)
            mask_topk    = (align >= topk_vals[:, -1:]) & in_box

            if n_gt > 1:
                conflict = mask_topk.sum(0) > 1
                if conflict.any():
                    best_gt  = align[:, conflict].argmax(0)
                    resolved = torch.zeros(n_gt, conflict.sum(), device=device)
                    resolved.scatter_(0, best_gt.unsqueeze(0), 1)
                    mask_topk[:, conflict] = resolved.bool()

            assigned_gt = mask_topk.float().argmax(0)
            is_fg       = mask_topk.any(0)

            if is_fg.any():
                align_fg   = align[:, is_fg]
                gt_iou_fg  = iou_mat[:, is_fg]
                max_align  = align_fg.amax(0, keepdim=True).clamp(min=self.eps)
                max_iou    = (gt_iou_fg * mask_topk[:, is_fg].float()).amax(0, keepdim=True)
                soft_score = (align_fg / max_align * max_iou).amax(0)

                pos_gt_idx = assigned_gt[is_fg]
                pos_labels = lbl_b[pos_gt_idx]
                soft_cls   = torch.zeros(is_fg.sum(), nc, device=device)
                soft_cls.scatter_(1, pos_labels.unsqueeze(1), soft_score.unsqueeze(1))

                fg_mask[b]               = is_fg
                target_gt_idx[b, is_fg]  = pos_gt_idx
                target_boxes[b, is_fg]   = gt_b[pos_gt_idx]
                target_scores[b, is_fg]  = soft_cls

        return target_boxes, target_scores, fg_mask, target_gt_idx


def _pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor,
                        eps: float = 1e-7) -> torch.Tensor:
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1    = ((boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])).clamp(0)
    a2    = ((boxes2[...,2]-boxes2[...,0])*(boxes2[...,3]-boxes2[...,1])).clamp(0)
    return inter / (a1 + a2 - inter + eps)


# ═══════════════════════════════════════════════════════════════════════════
# 9. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _ciou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor,
               eps: float = 1e-7) -> torch.Tensor:
    px1,py1,px2,py2 = pred_xyxy[:,0],pred_xyxy[:,1],pred_xyxy[:,2],pred_xyxy[:,3]
    tx1,ty1,tx2,ty2 = tgt_xyxy[:,0], tgt_xyxy[:,1], tgt_xyxy[:,2], tgt_xyxy[:,3]

    inter_w = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0)
    inter_h = (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
    inter   = inter_w * inter_h
    pw, ph  = (px2-px1).clamp(0), (py2-py1).clamp(0)
    tw, th  = (tx2-tx1).clamp(0), (ty2-ty1).clamp(0)
    union   = pw*ph + tw*th - inter + eps
    iou     = inter / union

    d2 = ((px1+px2)/2-(tx1+tx2)/2)**2 + ((py1+py2)/2-(ty1+ty2)/2)**2
    enc_w  = (torch.max(px2,tx2)-torch.min(px1,tx1)).clamp(0)
    enc_h  = (torch.max(py2,ty2)-torch.min(py1,ty1)).clamp(0)
    c2     = enc_w**2 + enc_h**2 + eps
    v      = (4/math.pi**2)*(torch.atan(tw/(th+eps))-torch.atan(pw/(ph+eps)))**2
    with torch.no_grad():
        alpha_v = v / (1 - iou + v + eps)
    return 1 - (iou - d2/c2 - alpha_v*v)


class DistributionFocalLoss(nn.Module):
    """DFL for one LTRB distance component."""
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl  = target.long()
        tr  = tl + 1
        wl  = tr.float() - target
        return (F.cross_entropy(pred_dist, tl, reduction="none") * wl
              + F.cross_entropy(pred_dist, tr, reduction="none") * (1.0 - wl))


class OrdinalRankingLoss(nn.Module):
    def __init__(self, ripeness_order: Optional[List[str]] = None,
                 margin: float = 0.2):
        super().__init__()
        self.ripeness_order = ripeness_order
        self.margin         = margin

    def _get_rank(self, label: int, class_names: List[str]) -> int:
        if self.ripeness_order is None:
            return label
        if label >= len(class_names):
            return label
        name = class_names[label].lower().strip()
        for rank, order_name in enumerate(self.ripeness_order):
            o = order_name.lower().strip()
            name_clean = name.replace("-", "").replace("_", "")
            o_clean    = o.replace("-", "").replace("_", "")
            if o in name or name in o or o_clean in name_clean or name_clean in o_clean:
                return rank
        return label

    def forward(self, maturity_scores: torch.Tensor,
                labels: torch.Tensor,
                class_names: List[str]) -> torch.Tensor:
        if maturity_scores.numel() < 2:
            return maturity_scores.sum() * 0.0

        ranks  = torch.tensor(
            [self._get_rank(int(l), class_names) for l in labels],
            dtype=torch.float32, device=maturity_scores.device)

        ri = ranks.unsqueeze(1);  rj = ranks.unsqueeze(0)
        si = maturity_scores.unsqueeze(1); sj = maturity_scores.unsqueeze(0)

        pair_mask = ri < rj
        if not pair_mask.any():
            return maturity_scores.sum() * 0.0

        loss = F.relu(self.margin + si - sj)[pair_mask]
        return loss.mean()


class AspectRatioPriorLoss(nn.Module):
    def __init__(self, min_aspect_ratio: float = 1.5,
                 gt_ratio_gate: float = 1.2):
        super().__init__()
        self.min_ratio    = min_aspect_ratio
        self.gt_ratio_gate = gt_ratio_gate

    def forward(self, pred_xyxy: torch.Tensor,
                gt_xyxy: Optional[torch.Tensor] = None,
                iou_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred_xyxy.numel() == 0:
            return pred_xyxy.sum() * 0.0

        w_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=1e-4)
        h_pred = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=1e-4)
        ratio_pred = h_pred / w_pred

        if gt_xyxy is not None and gt_xyxy.numel() > 0:
            w_gt = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=1e-4)
            h_gt = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=1e-4)
            gt_ratio = h_gt / w_gt
            gate_mask = gt_ratio >= self.gt_ratio_gate
            if not gate_mask.any():
                return pred_xyxy.sum() * 0.0
            ratio_pred   = ratio_pred[gate_mask]
            iou_weights  = iou_weights[gate_mask] if iou_weights is not None else None

        penalty = F.relu(self.min_ratio - ratio_pred)
        if iou_weights is not None:
            penalty = penalty * (1.0 - iou_weights.clamp(0, 1).detach())
        return penalty.mean()


class AgroNetLoss(nn.Module):
    """Combined loss for AgroNet."""
    def __init__(self, cfg: Config, num_classes: int,
                 class_names: Optional[List[str]] = None):
        super().__init__()
        self.num_classes      = num_classes
        self.reg_max          = cfg.reg_max
        self.img_size         = cfg.img_size
        self.box_weight       = cfg.box_weight
        self.cls_weight       = cfg.cls_weight
        self.dfl_weight       = cfg.dfl_weight
        self.ord_weight       = cfg.ord_weight
        self.asp_weight       = cfg.asp_weight
        self.cls_smooth       = cfg.cls_label_smooth
        self.cls_loss_reduction = cfg.cls_loss_reduction
        self.class_names      = class_names or []

        self.tal_assigner  = TaskAlignedLabelAssigner(
            cfg.tal_topk, cfg.tal_alpha, cfg.tal_beta)
        self.dfl_loss      = DistributionFocalLoss(cfg.reg_max)
        self.dfl_decoder   = DFLDistributionDecoder(cfg.reg_max)

        self.ordinal_loss  = OrdinalRankingLoss(
            ripeness_order=[
                "immature", "under-mature", "under_mature", "undermature",
                "mature", "good",
                "over-mature", "over_mature", "overmature",
            ],
            margin=0.2)
        self.aspect_loss   = AspectRatioPriorLoss(
            min_aspect_ratio=cfg.asp_min_ratio,
            gt_ratio_gate=1.2)

    @staticmethod
    def _collate_targets(targets: List[Dict[str, torch.Tensor]],
                          img_size: int, device: torch.device
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        B      = len(targets)
        max_gt = max(max(t["boxes"].shape[0] for t in targets), 1)
        boxes  = torch.full((B, max_gt, 4), 0.0,  dtype=torch.float32, device=device)
        labels = torch.full((B, max_gt),    -1,   dtype=torch.long,    device=device)
        for i, t in enumerate(targets):
            n = t["boxes"].shape[0]
            if n == 0:
                continue
            cx, cy, w, h = (t["boxes"][:, 0], t["boxes"][:, 1],
                             t["boxes"][:, 2], t["boxes"][:, 3])
            s  = float(img_size)
            boxes[i, :n]  = torch.stack(
                [(cx-w/2)*s, (cy-h/2)*s, (cx+w/2)*s, (cy+h/2)*s], -1).to(device)
            labels[i, :n] = t["labels"].to(device)
        return boxes, labels

    def _smooth_bce_targets(self, tgt_scores: torch.Tensor) -> torch.Tensor:
        eps = self.cls_smooth
        return tgt_scores * (1.0 - eps) + eps / self.num_classes

    def _compute_cls_loss(self, cls_flat: torch.Tensor, tgt_scores_smooth: torch.Tensor, n_pos: torch.Tensor) -> torch.Tensor:
        if self.cls_loss_reduction == "mean":
            return F.binary_cross_entropy_with_logits(
                cls_flat, tgt_scores_smooth, reduction="mean")
        else:
            return F.binary_cross_entropy_with_logits(
                cls_flat, tgt_scores_smooth, reduction="sum") / n_pos

    def forward(
        self,
        box_preds:      List[torch.Tensor],
        cls_preds:      List[torch.Tensor],
        maturity_preds: List[torch.Tensor],
        targets:        List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:

        device = box_preds[0].device
        B      = box_preds[0].shape[0]
        strides     = [8, 16, 32]
        feat_shapes = [(p.shape[2], p.shape[3]) for p in box_preds]
        anchor_pts, stride_t = build_anchor_grid(feat_shapes, strides, device)
        A  = anchor_pts.shape[0]
        nc = self.num_classes

        box_flat = torch.cat(
            [p.permute(0,2,3,1).reshape(B,-1,4*self.reg_max) for p in box_preds], 1)
        cls_flat = torch.cat(
            [p.permute(0,2,3,1).reshape(B,-1,nc) for p in cls_preds], 1)
        mat_flat = torch.cat(
            [p.permute(0,2,3,1).reshape(B,-1,1)  for p in maturity_preds], 1)

        with torch.no_grad():
            dfl_flat   = box_flat.view(B, A, 4, self.reg_max)
            ltrb_fm    = (dfl_flat.softmax(-1) *
                          torch.arange(self.reg_max, device=device, dtype=torch.float32)
                         ).sum(-1)
            anchor_img = anchor_pts * stride_t
            pred_xyxy  = ltrb_distances_to_xyxy_boxes(
                ltrb_fm.view(-1,4) * stride_t.repeat(B,1),
                anchor_img.unsqueeze(0).expand(B,-1,-1).reshape(-1,2)
            ).view(B, A, 4)

        gt_boxes, gt_labels = self._collate_targets(targets, self.img_size, device)
        tgt_boxes, tgt_scores, fg_mask, _ = self.tal_assigner(
            cls_flat.detach().sigmoid(), pred_xyxy.detach(),
            anchor_pts, stride_t, gt_boxes, gt_labels, self.img_size)
        n_pos = fg_mask.sum().clamp(min=1).float()

        tgt_scores_smooth = self._smooth_bce_targets(tgt_scores)
        loss_cls = self._compute_cls_loss(cls_flat, tgt_scores_smooth, n_pos)

        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        loss_ord = torch.tensor(0.0, device=device)
        loss_asp = torch.tensor(0.0, device=device)

        if fg_mask.any():
            pos_box_pred = box_flat[fg_mask]
            pos_stride   = stride_t.expand(B, -1, 1)[fg_mask]
            pos_anchor   = anchor_pts.unsqueeze(0).expand(B,-1,-1)[fg_mask]
            pos_mat_pred = mat_flat[fg_mask].squeeze(-1)
            pos_gt_xyxy  = tgt_boxes[fg_mask]

            dfl_pos  = pos_box_pred.view(-1, 4, self.reg_max)
            ltrb_pos = (dfl_pos.softmax(-1) *
                        torch.arange(self.reg_max, device=device, dtype=torch.float32)
                       ).sum(-1)
            pred_pos_xyxy = ltrb_distances_to_xyxy_boxes(
                ltrb_pos * pos_stride, pos_anchor * pos_stride)

            ciou     = _ciou_loss(pred_pos_xyxy, pos_gt_xyxy)
            loss_box = ciou.mean()

            with torch.no_grad():
                iou_w = _pairwise_iou_xyxy(
                    pred_pos_xyxy.unsqueeze(1), pos_gt_xyxy.unsqueeze(0)
                ).squeeze(0).clamp(0)
                iou_w = iou_w.diag() if iou_w.dim() == 2 else iou_w

            gt_ltrb = xyxy_boxes_to_ltrb_distances(
                pos_anchor, pos_gt_xyxy / pos_stride, self.reg_max)
            dfl_sum = sum(self.dfl_loss(dfl_pos[:, k, :], gt_ltrb[:, k]).mean()
                          for k in range(4))
            loss_dfl = dfl_sum / 4.0

            if self.class_names:
                pos_labels = tgt_scores[fg_mask].argmax(-1)
                loss_ord = self.ordinal_loss(
                    pos_mat_pred.sigmoid(), pos_labels, self.class_names)

            loss_asp = self.aspect_loss(pred_pos_xyxy,
                                        gt_xyxy=pos_gt_xyxy,
                                        iou_weights=iou_w)

        total = (self.box_weight * loss_box
               + self.cls_weight * loss_cls
               + self.dfl_weight * loss_dfl
               + self.ord_weight * loss_ord
               + self.asp_weight * loss_asp)

        return {"total": total, "box": loss_box, "cls": loss_cls,
                "dfl": loss_dfl, "ord": loss_ord, "asp": loss_asp}