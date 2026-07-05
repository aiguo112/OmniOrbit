#!/usr/bin/env python3
"""
Semantic segmentation baseline for the space ERP dataset (14 classes: background + 13).

Reads splits.json; for each frame uses image/<stem>.png and mask/<stem>.png. The RGB
palette mask is decoded to a class-index map via the fixed palette below.

Run (single GPU):
    conda activate pano360x
    pip install segmentation-models-pytorch
    CUDA_VISIBLE_DEVICES=1 python3 train_segment.py /home/arbi/infinigen/outputs/space_erp_final \
        --epochs 15 --batch 8 --img_h 512 --img_w 1024 --encoder resnet34

    # first quick look: cap the training set
    ... --max_train 8000 --epochs 10

Outputs into runs/segment/: best.pt, curves.png, overlays.png, per-class IoU printed.

Requires: torch torchvision segmentation-models-pytorch matplotlib numpy pillow
"""
import os, json, argparse, random, time, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

# fixed 14-class palette (index 0 = background/black, 1..13 = object classes)
PALETTE = {0: (0, 0, 0), 1: (220, 50, 50), 2: (50, 200, 80), 3: (50, 90, 220),
           4: (220, 200, 50), 5: (200, 100, 30), 6: (160, 60, 200), 7: (60, 200, 200),
           8: (240, 130, 200), 9: (250, 160, 40), 10: (120, 80, 220), 11: (30, 120, 255),
           12: (190, 190, 190), 13: (120, 100, 80)}
NAMES = ["background", "earth_obs_satellite", "space_telescope", "space_station",
         "deep_space_probe", "mars_mission", "comm_satellite", "cubesat", "astronaut_eva",
         "solar_heliophysics", "rocket_launch_vehicle", "planet", "moon", "asteroid"]
NUM = len(PALETTE)
P_ARR = np.array([PALETTE[i] for i in range(NUM)], np.int64)
P_PACK = (P_ARR[:, 0] << 16) | (P_ARR[:, 1] << 8) | P_ARR[:, 2]   # packed RGB per class

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--splits", default=None)
ap.add_argument("--img_h", type=int, default=512)
ap.add_argument("--img_w", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--encoder", default="resnet34")
ap.add_argument("--encoder_weights", default="imagenet", help="'imagenet' or 'none'")
ap.add_argument("--arch", default="unet", choices=["unet", "unetpp", "deeplabv3plus", "fpn", "pspnet"])
ap.add_argument("--tag", default=None, help="row label for the benchmark CSV")
ap.add_argument("--bench_csv", default="runs/segment_bench.csv")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max_train", type=int, default=0, help="cap train set (0 = all)")
ap.add_argument("--out", default="runs/segment")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
manifest = json.load(open(args.splits or os.path.join(args.root, "splits.json")))["splits"]
if args.max_train and len(manifest["train"]) > args.max_train:
    manifest["train"] = manifest["train"][:args.max_train]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
print(f"{NUM} classes, device={device}, train={len(manifest['train'])}")

def mask_to_label(rgb):
    rgb = rgb.astype(np.int64)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    label = np.zeros(packed.shape, np.uint8)
    for idx in range(NUM):                       # exact palette match; unmatched stays bg
        label[packed == P_PACK[idx]] = idx
    return label

from torchvision import transforms as T
jitter = T.ColorJitter(0.2, 0.2, 0.2)

class SegDS(Dataset):
    def __init__(self, root, stems, train):
        self.root, self.stems, self.train = root, stems, train
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(os.path.join(self.root, "image", stem + ".png")).convert("RGB")
        if self.train:
            img = jitter(img)
        img = img.resize((args.img_w, args.img_h), Image.BILINEAR)
        img = np.asarray(img, np.float32)
        m = np.asarray(Image.open(os.path.join(self.root, "mask", stem + ".png")).convert("RGB"))
        lab = mask_to_label(m)
        lab = np.asarray(Image.fromarray(lab).resize((args.img_w, args.img_h), Image.NEAREST))
        if self.train:                            # joint geometric aug (image + mask together)
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1]); lab = np.ascontiguousarray(lab[:, ::-1])
            off = random.randint(0, args.img_w - 1)         # ERP-valid longitude roll
            img = np.roll(img, off, axis=1); lab = np.roll(lab, off, axis=1)
        img = (img / 255.0 - MEAN) / STD
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        return img_t, torch.from_numpy(lab.astype(np.int64))

def loader(split, train, shuffle):
    return DataLoader(SegDS(args.root, manifest[split], train), batch_size=args.batch,
                      shuffle=shuffle, num_workers=args.workers, pin_memory=(device == "cuda"),
                      drop_last=shuffle)

train_dl = loader("train", True, True)
val_dl = loader("val", False, False)
test_dl = loader("test", False, False)

# median-frequency class weights from a sample of train masks
print("computing class weights from a sample...")
counts = np.zeros(NUM, np.int64)
for stem in random.sample(manifest["train"], min(300, len(manifest["train"]))):
    m = np.asarray(Image.open(os.path.join(args.root, "mask", stem + ".png")).convert("RGB"))
    lab = mask_to_label(m)
    counts += np.bincount(lab.reshape(-1), minlength=NUM)
freq = counts / max(counts.sum(), 1)
present = counts > 0
med = np.median(freq[present]) if present.any() else 1.0
weights = np.where(present, med / np.maximum(freq, 1e-8), 1.0)
weights = np.clip(weights, 0.1, 10.0).astype(np.float32)
print("class weights:", {NAMES[i]: round(float(weights[i]), 2) for i in range(NUM)})
w_t = torch.tensor(weights, device=device)

ARCH = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN, "pspnet": smp.PSPNet}
model = ARCH[args.arch](
    encoder_name=args.encoder,
    encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
    in_channels=3, classes=NUM).to(device)
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
ce = nn.CrossEntropyLoss(weight=w_t)
dice = smp.losses.DiceLoss(mode="multiclass")
use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def iou_from_cm(cm):
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    valid = union > 0
    return iou, valid

@torch.no_grad()
def evaluate(dl):
    model.eval()
    cm = np.zeros((NUM, NUM), np.int64)
    for x, y in dl:
        x = x.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            pred = model(x).argmax(1).cpu().numpy()
        y = y.numpy()
        k = (y.reshape(-1) * NUM + pred.reshape(-1))
        cm += np.bincount(k, minlength=NUM * NUM).reshape(NUM, NUM)
    iou, valid = iou_from_cm(cm)
    miou = iou[valid].mean()
    miou_fg = iou[1:][valid[1:]].mean() if valid[1:].any() else 0.0
    return miou, miou_fg, iou, valid

hist = {"loss": [], "miou": [], "miou_fg": []}
best = 0.0
for ep in range(1, args.epochs + 1):
    model.train(); t0 = time.time(); ls = 0; n = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            out = model(x)
            loss = ce(out, y) + dice(out, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ls += loss.item() * x.size(0); n += x.size(0)
    sched.step()
    miou, miou_fg, _, _ = evaluate(val_dl)
    hist["loss"].append(ls / n); hist["miou"].append(miou); hist["miou_fg"].append(miou_fg)
    print(f"epoch {ep:2d}/{args.epochs}  loss {ls/n:.3f}  val mIoU {miou:.3f}  "
          f"(fg {miou_fg:.3f})  ({time.time()-t0:.0f}s)")
    if miou >= best:
        best = miou
        torch.save({"model": model.state_dict(), "encoder": args.encoder, "names": NAMES}, os.path.join(args.out, "best.pt"))

# final test with best checkpoint
model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device)["model"])
miou, miou_fg, iou, valid = evaluate(test_dl)
print(f"\nTEST  mIoU {miou:.4f}   foreground mIoU {miou_fg:.4f}   (best val {best:.4f})")
print("per-class IoU:")
for i in range(NUM):
    print(f"  {NAMES[i]:24s} {iou[i]:.3f}" if valid[i] else f"  {NAMES[i]:24s}   n/a")

tag = args.tag or f"{args.arch}_{args.encoder}"
bench_header = ["tag", "arch", "encoder", "img_h", "img_w", "epochs", "params_m", "flops_g",
                "test_miou", "test_miou_fg"] + NAMES
bench_row = [tag, args.arch, args.encoder, args.img_h, args.img_w, args.epochs,
             f"{params_m:.4f}", f"{flops_g:.4f}" if not np.isnan(flops_g) else "",
             f"{miou:.4f}", f"{miou_fg:.4f}"]
bench_row += [f"{iou[i]:.4f}" if valid[i] else "" for i in range(NUM)]
bench_new = not os.path.exists(args.bench_csv)
with open(args.bench_csv, "a", newline="") as f:
    w = csv.writer(f)
    if bench_new:
        w.writerow(bench_header)
    w.writerow(bench_row)

# figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["loss"]); ax[0].set_title("Train loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["miou"], label="mIoU"); ax[1].plot(ep, hist["miou_fg"], label="fg mIoU")
    ax[1].set_title("Val mIoU"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "curves.png"), dpi=130); plt.close(fig)

    # qualitative overlays: image | GT | prediction
    def colorize(lab):
        out = np.zeros((*lab.shape, 3), np.uint8)
        for i in range(NUM):
            out[lab == i] = PALETTE[i]
        return out
    model.eval()
    picks = manifest["test"][:6]
    fig, ax = plt.subplots(len(picks), 3, figsize=(12, 2.2 * len(picks)))
    for r, stem in enumerate(picks):
        im = Image.open(os.path.join(args.root, "image", stem + ".png")).convert("RGB").resize((args.img_w, args.img_h))
        gt = mask_to_label(np.asarray(Image.open(os.path.join(args.root, "mask", stem + ".png")).convert("RGB")))
        gt = np.asarray(Image.fromarray(gt).resize((args.img_w, args.img_h), Image.NEAREST))
        x = ((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        with torch.no_grad(), torch.autocast(device_type=device, enabled=use_amp):
            pr = model(torch.from_numpy(x[None].copy()).float().to(device)).argmax(1)[0].cpu().numpy()
        for c, (title, im_show) in enumerate([("image", np.asarray(im)), ("ground truth", colorize(gt)), ("prediction", colorize(pr))]):
            ax[r, c].imshow(im_show); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(title)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "overlays.png"), dpi=110); plt.close(fig)
    print(f"\nsaved curves.png and overlays.png in {args.out}")
except Exception as e:
    print(f"[warn] figures skipped: {e}")
