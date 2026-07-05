#!/usr/bin/env python3
"""
Monocular depth-estimation baseline for the space ERP dataset.

Image/<stem>.png -> depth/<stem>.exr. Depth is object distance in scene units; empty
space is ~1e9 ("infinite") and is masked out. Because absolute scale is ambiguous for
isolated objects in space, this trains and evaluates SCALE-INVARIANTLY: the network
predicts log-depth with a SILog loss, and metrics are computed after per-image log-scale
alignment (standard practice for monocular depth, e.g. MiDaS/DPT).

Run:
    conda activate pano360x
    pip install opencv-python
    CUDA_VISIBLE_DEVICES=1 python3 train_depth.py space_erp_final \
        --epochs 20 --batch 8 --img_h 512 --img_w 1024 --encoder resnet34
    # quick:  add  --max_train 8000 --epochs 10

Metrics (valid pixels, scale-aligned): AbsRel, RMSE, delta<1.25. Outputs -> runs/depth/.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import json, argparse, random, time, csv
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
import segmentation_models_pytorch as smp

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--splits", default=None)
ap.add_argument("--img_h", type=int, default=512)
ap.add_argument("--img_w", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--encoder", default="resnet34")
ap.add_argument("--encoder_weights", default="imagenet")
ap.add_argument("--arch", default="unet", choices=["unet", "unetpp", "deeplabv3plus", "fpn", "pspnet"])
ap.add_argument("--tag", default=None, help="row label for the benchmark CSV")
ap.add_argument("--bench_csv", default="runs/depth_bench.csv")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--depth_max", type=float, default=1e6, help="depths above this = background (invalid)")
ap.add_argument("--out", default="runs/depth")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
manifest = json.load(open(args.splits or os.path.join(args.root, "splits.json")))["splits"]
if args.max_train and len(manifest["train"]) > args.max_train:
    manifest["train"] = manifest["train"][:args.max_train]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
print(f"device={device}, train={len(manifest['train'])}  (scale-invariant log-depth)")

def read_depth(path):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"could not read {path} (OpenEXR support?)")
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)

jitter = T.ColorJitter(0.2, 0.2, 0.2)

class DepthDS(Dataset):
    def __init__(self, root, stems, train):
        self.root, self.stems, self.train = root, stems, train
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(os.path.join(self.root, "image", stem + ".png")).convert("RGB")
        if self.train:
            img = jitter(img)
        img = np.asarray(img.resize((args.img_w, args.img_h), Image.BILINEAR), np.float32)
        d = read_depth(os.path.join(self.root, "depth", stem + ".exr"))
        d = cv2.resize(d, (args.img_w, args.img_h), interpolation=cv2.INTER_NEAREST)
        valid = (d > 1e-3) & (d < args.depth_max)
        d = np.where(valid, d, 1.0)                 # placeholder for invalid (ignored by mask)
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]; d = d[:, ::-1]; valid = valid[:, ::-1]
            off = random.randint(0, args.img_w - 1)
            img = np.roll(img, off, axis=1); d = np.roll(d, off, axis=1); valid = np.roll(valid, off, axis=1)
        img = (np.ascontiguousarray(img) / 255.0 - MEAN) / STD
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        d_t = torch.from_numpy(np.ascontiguousarray(d)).float()
        v_t = torch.from_numpy(np.ascontiguousarray(valid))
        return img_t, d_t, v_t

def loader(split, train, shuffle):
    return DataLoader(DepthDS(args.root, manifest[split], train), batch_size=args.batch,
                      shuffle=shuffle, num_workers=args.workers, pin_memory=(device == "cuda"),
                      drop_last=shuffle)

train_dl = loader("train", True, True)
val_dl = loader("val", False, False)
test_dl = loader("test", False, False)

ARCH = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN, "pspnet": smp.PSPNet}
model = ARCH[args.arch](
    encoder_name=args.encoder,
    encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
    in_channels=3, classes=1).to(device)
params_m = sum(p.numel() for p in model.parameters()) / 1e6
try:
    from ptflops import get_model_complexity_info
    with torch.no_grad():
        macs, _ = get_model_complexity_info(model, (3, args.img_h, args.img_w),
                                            as_strings=False, print_per_layer_stat=False, verbose=False)
    flops_g = 2 * macs / 1e9
except Exception:
    flops_g = float("nan")
print(f"[bench] arch={args.arch} enc={args.encoder} params={params_m:.1f}M flops={flops_g:.1f}G")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def silog_loss(pred_log, gt, valid, lam=0.85, alpha=10.0):
    g = pred_log.float() - torch.log(gt.clamp(min=1e-3))
    out = pred_log.sum() * 0.0
    tot = 0.0
    for b in range(pred_log.shape[0]):
        gb = g[b][valid[b]]
        if gb.numel() < 10:
            continue
        out = out + alpha * torch.sqrt((gb ** 2).mean() - lam * gb.mean() ** 2 + 1e-6)
        tot += 1
    return out / max(tot, 1)

@torch.no_grad()
def evaluate(dl):
    model.eval()
    absrel = rmse = d1 = nim = 0.0
    for x, d, v in dl:
        x = x.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            pl = model(x).squeeze(1).float().cpu()
        for b in range(pl.shape[0]):
            m = v[b].bool()
            if m.sum() < 10:
                continue
            plb = pl[b][m]; g = d[b][m]
            shift = (torch.log(g) - plb).mean()         # optimal log-scale alignment
            p = torch.exp(plb + shift)
            absrel += ((p - g).abs() / g).mean().item()
            rmse += torch.sqrt(((p - g) ** 2).mean()).item()
            ratio = torch.maximum(p / g, g / p)
            d1 += (ratio < 1.25).float().mean().item()
            nim += 1
    nim = max(nim, 1)
    return absrel / nim, rmse / nim, d1 / nim

hist = {"loss": [], "absrel": [], "d1": []}
best = 0.0
for ep in range(1, args.epochs + 1):
    model.train(); t0 = time.time(); ls = 0; cnt = 0
    for x, d, v in train_dl:
        x, d, v = x.to(device), d.to(device), v.to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            pl = model(x).squeeze(1)
            loss = silog_loss(pl, d, v)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ls += loss.item() * x.size(0); cnt += x.size(0)
    sched.step()
    absrel, rmse, d1 = evaluate(val_dl)
    hist["loss"].append(ls / cnt); hist["absrel"].append(absrel); hist["d1"].append(d1)
    print(f"epoch {ep:2d}/{args.epochs}  loss {ls/cnt:.4f}  val AbsRel {absrel:.3f}  "
          f"RMSE {rmse:.2f}  d1 {d1:.3f}  ({time.time()-t0:.0f}s)")
    if d1 >= best:
        best = d1
        torch.save({"model": model.state_dict(), "encoder": args.encoder}, os.path.join(args.out, "best.pt"))

model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device)["model"])
absrel, rmse, d1 = evaluate(test_dl)
print(f"\nTEST  AbsRel {absrel:.4f}   RMSE {rmse:.3f}   delta<1.25 {d1:.4f}   (best val d1 {best:.4f})")

tag = args.tag or f"{args.arch}_{args.encoder}"
bench_header = ["tag", "arch", "encoder", "img_h", "img_w", "epochs", "params_m", "flops_g",
                "test_absrel", "test_rmse", "test_d1"]
bench_row = [tag, args.arch, args.encoder, args.img_h, args.img_w, args.epochs,
             f"{params_m:.4f}", f"{flops_g:.4f}" if not np.isnan(flops_g) else "",
             f"{absrel:.4f}", f"{rmse:.4f}", f"{d1:.4f}"]
bench_new = not os.path.exists(args.bench_csv)
with open(args.bench_csv, "a", newline="") as f:
    w = csv.writer(f)
    if bench_new:
        w.writerow(bench_header)
    w.writerow(bench_row)

print("(scale-invariant: per-image log-scale aligned before metrics)")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["loss"]); ax[0].set_title("Train loss (SILog)"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["absrel"], label="AbsRel"); ax[1].plot(ep, hist["d1"], label="\u03b4<1.25")
    ax[1].set_title("Val depth metrics"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "curves.png"), dpi=130); plt.close(fig)

    model.eval()
    picks = manifest["test"][:6]
    fig, ax = plt.subplots(len(picks), 3, figsize=(12, 2.2 * len(picks)))
    for r, stem in enumerate(picks):
        im = Image.open(os.path.join(args.root, "image", stem + ".png")).convert("RGB").resize((args.img_w, args.img_h))
        d = read_depth(os.path.join(args.root, "depth", stem + ".exr"))
        d = cv2.resize(d, (args.img_w, args.img_h), interpolation=cv2.INTER_NEAREST)
        valid = (d > 1e-3) & (d < args.depth_max)
        x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        with torch.no_grad(), torch.autocast(device_type=device, enabled=use_amp):
            pl = model(torch.from_numpy(x[None].copy()).float().to(device)).squeeze().float().cpu().numpy()
        pr = np.exp(pl)
        if valid.any():
            shift = np.mean(np.log(d[valid]) - pl[valid]); pr = pr * np.exp(shift)
            vmin, vmax = d[valid].min(), d[valid].max()
        else:
            vmin, vmax = 0, 1
        def show(arr, mask):
            a = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
            rgb = (plt.cm.turbo(a)[..., :3] * 255).astype(np.uint8); rgb[~mask] = 0
            return rgb
        for c, (title, im_show) in enumerate([("image", np.asarray(im)),
                                              ("GT depth", show(d, valid)),
                                              ("pred depth (aligned)", show(pr, valid))]):
            ax[r, c].imshow(im_show); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(title)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "overlays.png"), dpi=110); plt.close(fig)
    print(f"saved curves.png and overlays.png in {args.out}")
except Exception as e:
    print(f"[warn] figures skipped: {e}")
