#!/usr/bin/env python3
"""
Pose-estimation baseline for the space ERP dataset — 3-DoF ATTITUDE (orientation).

Key ERP subtlety: a bearing-stripped object crop only reveals the object's orientation
RELATIVE to the viewing ray, while the logged quaternion is in the WORLD frame; the two
differ by the object's bearing. So world-frame attitude is NOT recoverable from the crop
alone. We therefore feed the object's bearing (azimuth/elevation, directly observable
from its position in the panorama) as an auxiliary input alongside the crop, and regress
the world-frame quaternion. Range is scale-ambiguous (see depth) and bearing is observed,
so attitude is the learnable core.

Run:
    conda activate pano360x
    CUDA_VISIBLE_DEVICES=1 python3 train_pose.py space_erp_final \
        --epochs 20 --batch 64 --arch resnet50 --workers 16
    # quick:  add  --max_train 8000 --epochs 12

Metric: geodesic angular error (deg) — median/mean, acc@10deg / acc@30deg. -> runs/pose/.
"""
import os, json, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--splits", default=None)
ap.add_argument("--crop", type=int, default=224)
ap.add_argument("--pad", type=float, default=0.25)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--arch", default="resnet50")
ap.add_argument("--encoder_weights", default="imagenet")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--no_bearing", action="store_true", help="ablation: drop the bearing input")
ap.add_argument("--out", default="runs/pose")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
manifest = json.load(open(args.splits or os.path.join(args.root, "splits.json")))["splits"]
if args.max_train and len(manifest["train"]) > args.max_train:
    manifest["train"] = manifest["train"][:args.max_train]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
print(f"device={device}, train={len(manifest['train'])}  "
      f"(attitude + {'NO ' if args.no_bearing else ''}bearing input)")

def meta_path(stem):
    for c in (os.path.join(args.root, "meta", stem + ".json"),
              os.path.join(args.root, "meta", os.path.basename(stem) + ".json")):
        if os.path.exists(c):
            return c
    return os.path.join(args.root, "meta", stem + ".json")

def load_pose(stem):
    m = json.load(open(meta_path(stem)))
    pc = m.get("primary_class")
    subj = None
    for o in m.get("objects", []):
        if o.get("class_id") == pc:
            subj = o; break
    if subj is None and m.get("objects"):
        subj = m["objects"][0]
    q = np.array(subj["rotation_quat"], np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    az = np.deg2rad(float(subj.get("azimuth", 0.0)))
    el = np.deg2rad(float(subj.get("elevation", 0.0)))
    bearing = np.array([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], np.float32)
    if args.no_bearing:
        bearing = np.zeros(4, np.float32)
    return q, bearing

from torchvision import transforms as T
jitter = T.ColorJitter(0.2, 0.2, 0.2)

def object_crop(img, mask):
    nb = mask.max(axis=2) > 0
    if not nb.any():
        return img
    W = mask.shape[1]
    if nb[:, 0].any() and nb[:, -1].any():
        img = np.roll(img, W // 2, axis=1); mask = np.roll(mask, W // 2, axis=1)
        nb = mask.max(axis=2) > 0
    ys, xs = np.where(nb)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    py, px = int(h * args.pad), int(w * args.pad)
    y0 = max(0, y0 - py); y1 = min(img.shape[0], y1 + py + 1)
    x0 = max(0, x0 - px); x1 = min(img.shape[1], x1 + px + 1)
    return img[y0:y1, x0:x1]

class PoseDS(Dataset):
    def __init__(self, root, stems, train):
        self.root, self.stems, self.train = root, stems, train
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        stem = self.stems[i]
        img = np.asarray(Image.open(os.path.join(self.root, "image", stem + ".png")).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(self.root, "mask", stem + ".png")).convert("RGB"))
        crop = object_crop(img, mask)
        pil = Image.fromarray(crop).resize((args.crop, args.crop), Image.BILINEAR)
        if self.train:
            pil = jitter(pil)
        a = (np.asarray(pil, np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(a.transpose(2, 0, 1).copy()).float()
        q, bearing = load_pose(stem)
        return x, torch.from_numpy(bearing), torch.from_numpy(q), stem.split("/")[0]

def loader(split, train, shuffle):
    return DataLoader(PoseDS(args.root, manifest[split], train), batch_size=args.batch,
                      shuffle=shuffle, num_workers=args.workers, pin_memory=(device == "cuda"),
                      drop_last=shuffle)

train_dl = loader("train", True, True)
val_dl = loader("val", False, False)
test_dl = loader("test", False, False)

class PoseNet(nn.Module):
    def __init__(self, arch, w):
        super().__init__()
        bb = getattr(torchvision.models, arch)(weights=w)
        self.fdim = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.head = nn.Sequential(nn.Linear(self.fdim + 4, 512), nn.ReLU(inplace=True),
                                  nn.Linear(512, 4))
    def forward(self, x, bearing):
        f = self.backbone(x)
        return self.head(torch.cat([f, bearing], dim=1))

w = None if args.encoder_weights == "none" else "IMAGENET1K_V2"
model = PoseNet(args.arch, w).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def quat_loss(pred, gt):
    pred = F.normalize(pred, dim=1)
    dot = (pred * gt).sum(1)
    return (1 - dot ** 2).mean()

def geodesic_deg(pred, gt):
    pred = F.normalize(pred, dim=1)
    dot = (pred * gt).sum(1).abs().clamp(max=1.0)
    return torch.rad2deg(2 * torch.arccos(dot))

@torch.no_grad()
def evaluate(dl, want_per_class=False):
    model.eval()
    errs = []; classes = []
    for x, b, q, cls in dl:
        x, b, q = x.to(device), b.to(device), q.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            pred = model(x, b).float()
        errs.append(geodesic_deg(pred, q).cpu()); classes += list(cls)
    errs = torch.cat(errs).numpy()
    out = dict(median=float(np.median(errs)), mean=float(errs.mean()),
               acc10=float((errs < 10).mean()), acc30=float((errs < 30).mean()))
    if want_per_class:
        out["errs"] = errs; out["classes"] = classes
    return out

hist = {"loss": [], "median": []}
best = 1e9
for ep in range(1, args.epochs + 1):
    model.train(); t0 = time.time(); ls = 0; n = 0
    for x, b, q, _ in train_dl:
        x, b, q = x.to(device), b.to(device), q.to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            loss = quat_loss(model(x, b), q)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ls += loss.item() * x.size(0); n += x.size(0)
    sched.step()
    v = evaluate(val_dl)
    hist["loss"].append(ls / n); hist["median"].append(v["median"])
    print(f"epoch {ep:2d}/{args.epochs}  loss {ls/n:.4f}  val median {v['median']:.1f}deg  "
          f"mean {v['mean']:.1f}  acc@10 {v['acc10']:.3f}  acc@30 {v['acc30']:.3f}  ({time.time()-t0:.0f}s)")
    if v["median"] <= best:
        best = v["median"]
        torch.save({"model": model.state_dict(), "arch": args.arch}, os.path.join(args.out, "best.pt"))

model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device)["model"])
t = evaluate(test_dl, want_per_class=True)
print(f"\nTEST  median {t['median']:.2f}deg  mean {t['mean']:.2f}deg  "
      f"acc@10 {t['acc10']:.4f}  acc@30 {t['acc30']:.4f}  (best val median {best:.2f})")
print("per-class median geodesic error (deg):")
errs, classes = t["errs"], np.array(t["classes"])
percls = {}
for c in sorted(set(classes)):
    percls[c] = float(np.median(errs[classes == c]))
    print(f"  {c:30s} {percls[c]:6.1f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["loss"]); ax[0].set_title("Train loss (1 - cos^2)"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["median"]); ax[1].set_title("Val median angular error (deg)"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "curves.png"), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = sorted(percls, key=lambda k: percls[k])
    ax[0].barh(range(len(names)), [percls[n] for n in names], color="#3c78c8")
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlabel("median angular error (deg)"); ax[0].set_title("Per-class attitude error")
    ax[1].hist(np.clip(errs, 0, 180), bins=40, color="#c86432")
    ax[1].set_xlabel("geodesic error (deg)"); ax[1].set_ylabel("count"); ax[1].set_title("Error distribution")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "errors.png"), dpi=130); plt.close(fig)
    print(f"saved curves.png and errors.png in {args.out}")
except Exception as e:
    print(f"[warn] figures skipped: {e}")
