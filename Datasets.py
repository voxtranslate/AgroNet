import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os
import cv2

# ═══════════════════════════════════════════════════════════════════════════
# 10. AUGMENTATION (Mosaic, Mixup, Copy-Paste)
# ═══════════════════════════════════════════════════════════════════════════

def _load_sample_raw(dataset: "OkraCocoDataset", idx: int,
                      scale_min: float, scale_max: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    info    = dataset.images[idx]
    src_idx = info["src_idx"]
    img     = Image.open(info["file_name"]).convert("RGB")
    w0, h0  = img.size
    boxes, labels = [], []
    for ann in dataset.cocos[src_idx].loadAnns(
            dataset.cocos[src_idx].getAnnIds(imgIds=[info["img_id"]], iscrowd=False)):
        if ann.get("iscrowd", 0) or ann["bbox"][2] <= 1 or ann["bbox"][3] <= 1:
            continue
        x, y, bw, bh = ann["bbox"]
        boxes.append([min(max((x+bw/2)/w0,0.),1.), min(max((y+bh/2)/h0,0.),1.),
                      min(max(bw/w0,1e-6),1.),  min(max(bh/h0,1e-6),1.)])
        labels.append(dataset.local_to_global[src_idx][ann["category_id"]])
    scale = random.uniform(scale_min, scale_max)
    img   = img.resize((max(1,int(w0*scale)), max(1,int(h0*scale))), _RESAMPLE_BILINEAR)
    return (np.array(img, dtype=np.uint8),
            np.array(boxes,  np.float32) if boxes  else np.zeros((0,4), np.float32),
            np.array(labels, np.int64)   if labels else np.zeros((0,),  np.int64))


def build_mosaic4(dataset: "OkraCocoDataset", idx: int, img_size: int,
                  scale_min: float, scale_max: float,
                  color_jitter: transforms.ColorJitter
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n    = len(dataset.images)
    idxs = [idx] + random.choices(range(n), k=3)
    cx   = random.randint(img_size//4, 3*img_size//4)
    cy   = random.randint(img_size//4, 3*img_size//4)
    placements = [(0,0,cx,cy),(cx,0,img_size,cy),(0,cy,cx,img_size),(cx,cy,img_size,img_size)]
    canvas     = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    all_boxes, all_labels = [], []

    for i, (x1,y1,x2,y2) in enumerate(placements):
        img_np, boxes, labels = _load_sample_raw(dataset, idxs[i], scale_min, scale_max)
        if random.random() < 0.5:
            img_np = np.array(color_jitter(Image.fromarray(img_np)), dtype=np.uint8)
        h, w = img_np.shape[:2]
        tw, th = x2-x1, y2-y1
        scale  = min(tw/w, th/h)
        rw, rh = max(1,int(w*scale)), max(1,int(h*scale))
        tile   = np.array(Image.fromarray(img_np).resize((rw,rh), _RESAMPLE_BILINEAR))
        ox = [x2-rw, x1, x2-rw, x1][i]
        oy = [y2-rh, y2-rh, y1, y1][i]
        cx1,cy1 = max(ox,0),max(oy,0)
        cx2,cy2 = min(ox+rw,img_size),min(oy+rh,img_size)
        if cx2>cx1 and cy2>cy1:
            canvas[cy1:cy2,cx1:cx2] = tile[cy1-oy:cy2-oy, cx1-ox:cx2-ox]
        if boxes.shape[0] == 0:
            continue
        bx1 = np.clip((boxes[:,0]-boxes[:,2]/2)*rw+ox, 0, img_size)
        by1 = np.clip((boxes[:,1]-boxes[:,3]/2)*rh+oy, 0, img_size)
        bx2 = np.clip((boxes[:,0]+boxes[:,2]/2)*rw+ox, 0, img_size)
        by2 = np.clip((boxes[:,1]+boxes[:,3]/2)*rh+oy, 0, img_size)
        valid = (bx2-bx1>2)&(by2-by1>2)
        if not valid.any(): continue
        bx1,by1,bx2,by2 = bx1[valid],by1[valid],bx2[valid],by2[valid]
        all_boxes.append(np.stack(
            [((bx1+bx2)/2)/img_size, ((by1+by2)/2)/img_size,
             (bx2-bx1)/img_size,     (by2-by1)/img_size], axis=1))
        all_labels.append(labels[valid])

    final_boxes  = np.concatenate(all_boxes,  0) if all_boxes  else np.zeros((0,4),np.float32)
    final_labels = np.concatenate(all_labels, 0) if all_labels else np.zeros((0,), np.int64)
    return canvas, final_boxes, final_labels


def apply_mixup(img1, boxes1, labels1, img2, boxes2, labels2, alpha=0.5):
    r    = random.betavariate(alpha, alpha)
    if img2.shape[:2] != img1.shape[:2]:
        img2 = np.array(Image.fromarray(img2).resize(
            (img1.shape[1], img1.shape[0]), _RESAMPLE_BILINEAR))
    blended = (r*img1.astype(np.float32)+(1-r)*img2.astype(np.float32)
               ).clip(0,255).astype(np.uint8)
    def _cat(a,b): return (np.concatenate([a,b],0) if a.shape[0] and b.shape[0]
                           else (a if a.shape[0] else b))
    return blended, _cat(boxes1, boxes2), _cat(labels1, labels2)


def apply_copy_paste(img, boxes, labels, donor_img, donor_boxes, donor_labels,
                     prob=0.5):
    if donor_boxes.shape[0] == 0:
        return img, boxes, labels
    H, W = img.shape[:2]
    new_boxes, new_labels = list(boxes), list(labels)
    for i in range(donor_boxes.shape[0]):
        if random.random() > prob: continue
        cx,cy,bw,bh = donor_boxes[i]
        dH,dW = donor_img.shape[:2]
        x1=max(0,int((cx-bw/2)*dW)); x2=min(dW,int((cx+bw/2)*dW))
        y1=max(0,int((cy-bh/2)*dH)); y2=min(dH,int((cy+bh/2)*dH))
        if x2-x1<2 or y2-y1<2: continue
        crop = donor_img[y1:y2,x1:x2]
        pw,ph = x2-x1,y2-y1
        if pw>=W or ph>=H: continue
        px=random.randint(0,W-pw); py=random.randint(0,H-ph)
        img[py:py+ph,px:px+pw] = crop
        new_boxes.append([(px+pw/2)/W,(py+ph/2)/H,pw/W,ph/H])
        new_labels.append(int(donor_labels[i]))
    return (img,
            np.array(new_boxes, np.float32) if new_boxes else np.zeros((0,4),np.float32),
            np.array(new_labels,np.int64)   if new_labels else np.zeros((0,),np.int64))


# ═══════════════════════════════════════════════════════════════════════════
# 11. DATASET
# ═══════════════════════════════════════════════════════════════════════════

class OkraCocoDataset(Dataset):
    """Multi-source COCO-style dataset with unified category space."""

    def __init__(self, sources: List[Tuple[str, str]], img_size: int = 640,
                 augment: bool = False, cfg: Optional[Config] = None):
        super().__init__()
        self.img_size      = img_size
        self.augment       = augment
        self.cfg           = cfg
        self.current_epoch = 0
        self.color_jitter  = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        self.resize        = transforms.Resize((img_size, img_size))
        self.to_tensor     = transforms.ToTensor()
        self.cocos:             List[COCO]             = []
        self.cat_name_to_gid: Dict[str, int]        = {}
        self.gid_to_cat_name: Dict[int, str]        = {}
        self.local_to_global: List[Dict[int, int]] = []
        self.images:          List[Dict]           = []
        self._build_index(sources)

    def _build_index(self, sources):
        for img_root, ann_path in sources:
            coco = COCO(ann_path)
            self.cocos.append(coco)
            l2g: Dict[int, int] = {}
            for cat in coco.loadCats(coco.getCatIds()):
                name = cat["name"]
                if name not in self.cat_name_to_gid:
                    gid = len(self.cat_name_to_gid)
                    self.cat_name_to_gid[name] = gid
                    self.gid_to_cat_name[gid]  = name
                l2g[cat["id"]] = self.cat_name_to_gid[name]
            self.local_to_global.append(l2g)
        for src_idx, (img_root, _) in enumerate(sources):
            for img_id in self.cocos[src_idx].getImgIds():
                info = self.cocos[src_idx].loadImgs(img_id)[0]
                self.images.append({
                    "src_idx":   src_idx, "img_id": img_id,
                    "file_name": os.path.join(img_root, info["file_name"]),
                    "width": info["width"], "height": info["height"],
                })

    @property
    def num_classes(self) -> int:
        return len(self.cat_name_to_gid)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        cfg = self.cfg
        use_mosaic = (self.augment and cfg is not None
                      and random.random() < cfg.mosaic_prob
                      and self.current_epoch < cfg.epochs - cfg.mosaic_off_epoch)

        if use_mosaic:
            img_np, boxes_np, labels_np = build_mosaic4(
                self, idx, self.img_size,
                cfg.scale_min, cfg.scale_max, self.color_jitter)
            if random.random() < cfg.copy_paste_prob and len(self.images) > 1:
                d_img, d_boxes, d_labels = _load_sample_raw(
                    self, random.randint(0,len(self.images)-1),
                    cfg.scale_min, cfg.scale_max)
                if d_img.shape[:2] != (self.img_size,)*2:
                    d_img = np.array(Image.fromarray(d_img).resize(
                        (self.img_size,)*2, _RESAMPLE_BILINEAR))
                img_np, boxes_np, labels_np = apply_copy_paste(
                    img_np, boxes_np, labels_np, d_img, d_boxes, d_labels, 0.5)
            if random.random() < cfg.mixup_prob:
                m_img, m_boxes, m_labels = build_mosaic4(
                    self, random.randint(0,len(self.images)-1),
                    self.img_size, cfg.scale_min, cfg.scale_max, self.color_jitter)
                img_np, boxes_np, labels_np = apply_mixup(
                    img_np, boxes_np, labels_np, m_img, m_boxes, m_labels)
            img    = Image.fromarray(img_np)
            boxes  = torch.from_numpy(boxes_np)
            labels = torch.from_numpy(labels_np)
            if random.random() < 0.5:
                img = img.transpose(_FLIP_LR)
                if boxes.numel(): boxes[:, 0] = 1.0 - boxes[:, 0]
            if random.random() < 0.3:
                img = img.transpose(_FLIP_TB)
                if boxes.numel(): boxes[:, 1] = 1.0 - boxes[:, 1]
        else:
            info    = self.images[idx]
            src_idx = info["src_idx"]
            img_id  = info["img_id"]
            img     = Image.open(info["file_name"]).convert("RGB")
            w0, h0  = img.size
            box_list, lbl_list = [], []
            for ann in self.cocos[src_idx].loadAnns(
                    self.cocos[src_idx].getAnnIds(imgIds=[img_id], iscrowd=False)):
                if ann.get("iscrowd",0) or ann["bbox"][2]<=1 or ann["bbox"][3]<=1:
                    continue
                x,y,bw,bh = ann["bbox"]
                box_list.append([min(max((x+bw/2)/w0,0.),1.),
                                  min(max((y+bh/2)/h0,0.),1.),
                                  min(max(bw/w0,1e-6),1.),
                                  min(max(bh/h0,1e-6),1.)])
                lbl_list.append(self.local_to_global[src_idx][ann["category_id"]])
            boxes  = (torch.tensor(box_list, dtype=torch.float32)
                      if box_list  else torch.zeros((0,4)))
            labels = (torch.tensor(lbl_list, dtype=torch.long)
                      if lbl_list else torch.zeros((0,), dtype=torch.long))
            if self.augment:
                if random.random() < 0.5: img = self.color_jitter(img)
                if random.random() < 0.5:
                    img = img.transpose(_FLIP_LR)
                    if boxes.numel(): boxes[:,0] = 1.0 - boxes[:,0]
                if random.random() < 0.3:
                    img = img.transpose(_FLIP_TB)
                    if boxes.numel(): boxes[:,1] = 1.0 - boxes[:,1]

        img_id_val = self.images[idx]["img_id"]
        return (self.to_tensor(self.resize(img)),
                {"boxes":    boxes.float(),
                 "labels":   labels.long(),
                 "image_id": torch.tensor([img_id_val], dtype=torch.int64)})


def okra_collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def make_train_val_test_splits(
    dataset: OkraCocoDataset,
    train_ratio: float, val_ratio: float, seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    n   = len(dataset)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(n * train_ratio)
    n_vl = int(n * val_ratio)

    def _copy(augment: bool) -> OkraCocoDataset:
        ds               = copy.copy(dataset)
        ds.augment       = augment
        ds.current_epoch = 0
        ds.cfg           = dataset.cfg if augment else None
        return ds

    return (Subset(_copy(True),  idx[:n_tr]),
            Subset(_copy(False), idx[n_tr:n_tr+n_vl]),
            Subset(_copy(False), idx[n_tr+n_vl:]))