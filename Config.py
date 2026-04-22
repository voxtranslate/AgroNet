import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All hyper-parameters in one place."""

    train_sources: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("/kaggle/input/datasets/perrinlitetli/okra-1/images/default",
         "/kaggle/input/datasets/perrinlitetli/okra-1/annotations/instances_default.json"),
        ("/kaggle/input/datasets/perrinlitetli/okra-2/images/default",
         "/kaggle/input/datasets/perrinlitetli/okra-2/annotations/instances_default.json"),
    ])

    train_ratio: float = 0.80
    val_ratio:   float = 0.10
    test_ratio:  float = 0.10
    split_seed:  int   = 42

    img_size:            int  = 640
    pretrained_backbone: bool = True
    reg_max:             int  = 16

    batch_size:           int   = 16
    num_workers:          int   = 4
    epochs:               int   = 1000
    warmup_epochs:        int   = 5
    lr:                   float = 1e-3
    lr_min:               float = 8e-6
    weight_decay:         float = 3e-4
    use_amp:              bool  = True
    backbone_lr_factor:   float = 0.20
    freeze_backbone_epochs: int = 5

    mosaic_prob:      float = 0.80
    mosaic_off_epoch: int   = 60
    mixup_prob:       float = 0.15
    scale_min:        float = 0.70
    scale_max:        float = 1.50
    copy_paste_prob:  float = 0.30

    box_weight: float = 7.5
    cls_weight: float = 15.0
    dfl_weight: float = 2.5
    ord_weight: float = 0.3
    asp_weight: float = 0.2

    cls_loss_reduction: str   = "mean"
    cls_label_smooth:   float = 0.1

    asp_min_ratio: float = 1.5

    tal_alpha:  float = 0.5
    tal_beta:   float = 6.0
    tal_topk:   int   = 10

    conf_threshold:      float = 0.45
    eval_conf_threshold: float = 0.001
    softnms_sigma:       float = 0.3
    softnms_score_thr:   float = 0.001
    iou_threshold:       float = 0.50
    max_det:             int   = 300
    dedup_iou_thr:       float = 0.60
    dedup_centre_px:     float = 35.0

    map_iou_50:    float       = 0.50
    map_iou_range: List[float] = field(default_factory=lambda:
                                       [round(t, 2) for t in
                                        list(np.arange(0.50, 1.00, 0.05))])

    log_dir:     str = "runs/agronet"
    results_dir: str = "results/agronet"

    device: str = "cuda"

    maturity_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "immature":    (0,   210, 90),
        "under-mature":(0,   210, 90),
        "under_mature":(0,   210, 90),
        "mature":      (60,  200, 255),
        "good":        (60,  200, 255),
        "over-mature": (255, 120, 0),
        "over_mature": (255, 120, 0),
        "diseased":    (220, 40,  60),
        "damaged":     (220, 40,  60),
    })
    default_bbox_color: Tuple[int, int, int] = (200, 200, 200)