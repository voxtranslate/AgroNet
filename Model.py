import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torchvision import models, transforms
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any

# =============================================
# MODULAR MODEL COMPONENTS
# =============================================
class ExponentialMovingAverageWeights:
    """Stabilises training by maintaining an EMA copy of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000,
                 updates: int = 0):
        self.ema     = copy.deepcopy(model).eval()
        self.updates = updates
        self.decay   = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d   = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()


# ═══════════════════════════════════════════════════════════════════════════
# 3. BACKBONE – ResNet-50 (ImageNet pretrained)
# ═══════════════════════════════════════════════════════════════════════════

class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet-50 backbone producing three multi-scale feature maps:
      C3  stride  8   –  512 channels
      C4  stride 16   – 1024 channels
      C5  stride 32   – 2048 channels
    """
    _CONV1_L2_LO: float = 6.0
    _CONV1_L2_HI: float = 12.0

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
            tqdm.write(f"[Backbone] Loading ResNet-50 weights: {weights}")
        else:
            weights = None
            tqdm.write("[Backbone] Training from scratch (pretrained=False)")

        resnet       = tv_models.resnet50(weights=weights)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

        if pretrained:
            self._verify_pretrained_weights()

    def _verify_pretrained_weights(self) -> None:
        with torch.no_grad():
            l2 = self.conv1.weight.norm().item()
        if not (self._CONV1_L2_LO <= l2 <= self._CONV1_L2_HI):
            raise RuntimeError(
                f"[Backbone] Pretrained check FAILED – conv1 L2={l2:.4f} "
                f"not in [{self._CONV1_L2_LO}, {self._CONV1_L2_HI}]")
        rm = self.bn1.running_mean.norm().item()
        if rm == 0.0:
            raise RuntimeError(
                "[Backbone] Pretrained check FAILED – bn1.running_mean is zero")
        tqdm.write(f"[Backbone] ✓ Pretrained verified  conv1_L2={l2:.3f}  bn1_rm={rm:.4f}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x  = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# ═══════════════════════════════════════════════════════════════════════════
# 4. BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

class ConvBnSilu(nn.Module):
    """Conv2d → BatchNorm → SiLU"""
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1,
                 p: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CbamAttentionGate(nn.Module):
    """CBAM-style channel + spatial attention gate."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden            = max(1, channels // reduction)
        self.avg_pool     = nn.AdaptiveAvgPool2d(1)
        self.channel_fc1  = nn.Conv2d(channels, hidden, 1)
        self.channel_fc2  = nn.Conv2d(hidden, channels, 1)
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid      = nn.Sigmoid()
        self.relu         = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.sigmoid(self.channel_fc2(
             self.relu(self.channel_fc1(self.avg_pool(x)))))
        x  = x * ca
        sa = self.sigmoid(self.spatial_conv(
             torch.cat([x.max(1, keepdim=True)[0],
                        x.mean(1, keepdim=True)], 1)))
        return x * sa


class QKVCrossScaleAttention(nn.Module):
    """Query-Key-Value cross-scale attention."""
    def __init__(self, dim_high: int, dim_low: int, dim_attn: int = 64):
        super().__init__()
        self.q_proj   = nn.Conv2d(dim_high, dim_attn, 1, bias=False)
        self.k_proj   = nn.Conv2d(dim_low,  dim_attn, 1, bias=False)
        self.v_proj   = nn.Conv2d(dim_low,  dim_attn, 1, bias=False)
        self.out_proj = nn.Conv2d(dim_attn, dim_low,  1, bias=False)
        self.scale    = dim_attn ** -0.5

    def forward(self, feat_high: torch.Tensor,
                feat_low: torch.Tensor) -> torch.Tensor:
        B, _, Hh, Wh = feat_high.shape
        _, _, Hl, Wl = feat_low.shape
        q    = self.q_proj(feat_high).view(B, -1, Hh * Wh).permute(0, 2, 1)
        k    = self.k_proj(feat_low).view(B, -1, Hl * Wl)
        v    = self.v_proj(feat_low).view(B, -1, Hl * Wl).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(B, -1, Hh, Wh)
        out  = F.interpolate(self.out_proj(out), size=(Hl, Wl), mode="nearest")
        return feat_low + out


class ChannelSelectiveFusionCalib(nn.Module):
    """Channel-Selective Fusion Token calibration."""
    def __init__(self, channels_high: int, channels_low: int, reduction: int = 16):
        super().__init__()
        self.align_high = nn.Conv2d(channels_high, channels_low, 1, bias=False)
        hidden          = max(1, channels_low // reduction)
        self.gating_mlp = nn.Sequential(
            nn.Linear(channels_low, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat_high: torch.Tensor, feat_low: torch.Tensor) -> torch.Tensor:
        f_high = self.align_high(feat_high)
        g      = (F.adaptive_avg_pool2d(f_high, 1).view(f_high.size(0), -1)
                  + F.adaptive_avg_pool2d(feat_low, 1).view(feat_low.size(0), -1))
        w      = self.softmax(self.gating_mlp(g))
        w_h    = w[:, 0].view(-1, 1, 1, 1)
        w_l    = w[:, 1].view(-1, 1, 1, 1)
        feat_high = F.interpolate(f_high, size=feat_low.shape[-2:], mode="nearest")
        return (w_h * feat_high + w_l * feat_low)


class GatedSkipFusionBlock(nn.Module):
    """Learnable-gated raw-image skip connection."""
    def __init__(self, backbone_ch: int, raw_ch: int = 3):
        super().__init__()
        self.raw_proj   = ConvBnSilu(raw_ch, backbone_ch, k=1, s=1, p=0)
        self.gate_conv  = nn.Conv2d(backbone_ch * 2, backbone_ch, 1, bias=True)
        self.gate_act   = nn.Sigmoid()

    def forward(self, backbone_feat: torch.Tensor,
                raw_patch: torch.Tensor) -> torch.Tensor:
        raw_proj = self.raw_proj(
            F.interpolate(raw_patch, size=backbone_feat.shape[2:],
                          mode="bilinear", align_corners=False))
        gate = self.gate_act(
            self.gate_conv(torch.cat([backbone_feat, raw_proj], dim=1)))
        return gate * backbone_feat + (1 - gate) * raw_proj


class SpatialPyramidPoolingFast(nn.Module):
    """SPPF – Spatial Pyramid Pooling Fast."""
    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        c_ = c_in // 2
        self.reduce  = ConvBnSilu(c_in,   c_,      k=1, s=1, p=0)
        self.project = ConvBnSilu(c_ * 4, c_out,   k=1, s=1, p=0)
        self.pool    = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.reduce(x)
        y1 = self.pool(x);  y2 = self.pool(y1);  y3 = self.pool(y2)
        return self.project(torch.cat([x, y1, y2, y3], dim=1))


class CrossScaleFPNFusionBlock(nn.Module):
    """FPN merge block with cross-scale fusion."""
    def __init__(self, c_high: int, c_low: int, c_out: int,
                 mode: str = "qkv_cross", use_post_cbam: bool = True):
        super().__init__()
        assert mode in ("qkv_cross", "csft_calib")
        self.proj_high = ConvBnSilu(c_high, c_out, k=1, s=1, p=0)
        self.cross_fuser = (
            QKVCrossScaleAttention(dim_high=c_out, dim_low=c_low)
            if mode == "qkv_cross"
            else ChannelSelectiveFusionCalib(channels_high=c_out, channels_low=c_low))
        self.merge_conv   = ConvBnSilu(c_out + c_low, c_out, k=3, s=1, p=1)
        self.use_post_cbam = use_post_cbam
        if use_post_cbam:
            self.post_cbam = CbamAttentionGate(c_out)

    def forward(self, feat_high: torch.Tensor,
                feat_low: torch.Tensor) -> torch.Tensor:
        feat_high    = self.proj_high(feat_high)
        feat_low_cal = self.cross_fuser(feat_high, feat_low)
        feat_up      = F.interpolate(feat_high, size=feat_low.shape[-2:], mode="nearest")
        x = self.merge_conv(torch.cat([feat_up, feat_low_cal], dim=1))
        if self.use_post_cbam:
            x = self.post_cbam(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 5. DETECTION HEADS
# ═══════════════════════════════════════════════════════════════════════════

class DFLDistributionDecoder(nn.Module):
    """Decodes DFL output at inference time."""
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.softmax(-1) @ self.proj


class MaturityAwareDecoupledHead(nn.Module):
    """Three branches: box, class, maturity."""
    def __init__(self, c_in: int, num_classes: int, reg_max: int = 16,
                 drop_path: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.drop_path_p = drop_path

        c_box = max(16, c_in // 4, reg_max * 4)
        c_cls = max(c_in, min(num_classes, 100))
        c_mat = max(16, c_in // 8)

        self.box_branch = nn.Sequential(
            ConvBnSilu(c_in, c_box, 3, 1, 1),
            ConvBnSilu(c_box, c_box, 3, 1, 1),
            nn.Conv2d(c_box, 4 * reg_max, 1),
        )
        self.cls_branch = nn.Sequential(
            ConvBnSilu(c_in, c_cls, 3, 1, 1),
            ConvBnSilu(c_cls, c_cls, 3, 1, 1),
            nn.Conv2d(c_cls, num_classes, 1),
        )
        self.maturity_branch = nn.Sequential(
            ConvBnSilu(c_in, c_mat, 3, 1, 1),
            nn.Conv2d(c_mat, 1, 1),
        )
        self._init_biases()

    def _init_biases(self) -> None:
        prior  = 0.01
        b_init = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_branch[-1].bias, b_init)
        nn.init.constant_(self.box_branch[-1].bias, 1.0)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_p == 0.0:
            return x
        keep = 1.0 - self.drop_path_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device) < keep
        return x * mask / keep

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_dp = self._drop_path(x)
        return self.box_branch(x_dp), self.cls_branch(x_dp), self.maturity_branch(x_dp)


# ═══════════════════════════════════════════════════════════════════════════
# 6. FULL DETECTOR – AgroNet
# ═══════════════════════════════════════════════════════════════════════════

class AgroNet(nn.Module):
    """
    AgroNet: ResNet-50 + GatedSkipFusion + SPPF + FPN-PAN + MaturityAwareDecoupledHead.
    """
    def __init__(self, num_classes: int, reg_max: int = 16,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.strides     = [8, 16, 32]

        self.backbone = ResNet50FeatureExtractor(pretrained=pretrained_backbone)

        c3, c4, c5 = 512, 1024, 2048

        self.gated_skip_c3 = GatedSkipFusionBlock(backbone_ch=c3)
        self.gated_skip_c4 = GatedSkipFusionBlock(backbone_ch=c4)
        self.gated_skip_c5 = GatedSkipFusionBlock(backbone_ch=c5)

        self.sppf = SpatialPyramidPoolingFast(c5, c5)

        self.fpn_project_p5 = ConvBnSilu(c5, 512, k=1, s=1, p=0)
        self.fpn_fuse_p4    = CrossScaleFPNFusionBlock(c_high=512, c_low=c4, c_out=256, mode="qkv_cross")
        self.fpn_fuse_p3    = CrossScaleFPNFusionBlock(c_high=256, c_low=c3, c_out=128, mode="csft_calib")

        self.panet_downsample_p3 = ConvBnSilu(128, 128, k=3, s=2, p=1)
        self.panet_fuse_p4       = ConvBnSilu(128 + 256, 256, k=3, s=1, p=1)
        self.panet_downsample_p4 = ConvBnSilu(256, 256, k=3, s=2, p=1)
        self.panet_fuse_p5       = ConvBnSilu(256 + 512, 512, k=3, s=1, p=1)

        self.head_p3 = MaturityAwareDecoupledHead(128, num_classes, reg_max, drop_path=0.1)
        self.head_p4 = MaturityAwareDecoupledHead(256, num_classes, reg_max, drop_path=0.1)
        self.head_p5 = MaturityAwareDecoupledHead(512, num_classes, reg_max, drop_path=0.1)

        self.dfl_decoder = DFLDistributionDecoder(reg_max)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        c3, c4, c5 = self.backbone(x)

        c3 = self.gated_skip_c3(c3, x)
        c4 = self.gated_skip_c4(c4, x)
        c5 = self.gated_skip_c5(c5, x)

        c5 = self.sppf(c5)

        p5 = self.fpn_project_p5(c5)
        p4 = self.fpn_fuse_p4(p5, c4)
        p3 = self.fpn_fuse_p3(p4, c3)

        n3 = p3
        n4 = self.panet_fuse_p4(torch.cat([self.panet_downsample_p3(n3), p4], dim=1))
        n5 = self.panet_fuse_p5(torch.cat([self.panet_downsample_p4(n4), p5], dim=1))

        b3, c3_, m3 = self.head_p3(n3)
        b4, c4_, m4 = self.head_p4(n4)
        b5, c5_, m5 = self.head_p5(n5)

        return [b3, b4, b5], [c3_, c4_, c5_], [m3, m4, m5]