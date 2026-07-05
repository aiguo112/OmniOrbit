#!/usr/bin/env python3
"""
13-class classification baseline for the space ERP dataset.

Reads splits.json (from split_dataset.py); the label is the subject class folder.

Run (single GPU):
    CUDA_VISIBLE_DEVICES=1 python3 train_classify.py /home/arbi/infinigen/outputs/space_erp_final \
        --epochs 8 --batch 32 --arch resnet50

Outputs into runs/classify/: best.pt, classes.json, curves.png, confusion_matrix.png

Requires: torch torchvision matplotlib  (install the CUDA build of torch for your system)
"""
import os, json, argparse, random, time, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--splits", default=None, help="path to splits.json (default <root>/splits.json)")
ap.add_argument("--img_h", type=int, default=256)
ap.add_argument("--img_w", type=int, default=512)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--arch", default="resnet50",
                choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large",
                         "convnext_tiny", "vit_b_16", "swin_t"])
ap.add_argument("--tag", default=None, help="row label for the benchmark CSV")
ap.add_argument("--bench_csv", default="runs/classify_bench.csv")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", default="runs/classify")
ap.add_argument("--no_pretrained", action="store_true")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
splits_path = args.splits or os.path.join(args.root, "splits.json")
manifest = json.load(open(splits_path))["splits"]

# class list from all stems (folder prefix before '/')
classes = sorted({s.split("/")[0] for part in manifest.values() for s in part})
cls2idx = {c: i for i, c in enumerate(classes)}
json.dump(classes, open(os.path.join(args.out, "classes.json"), "w"), indent=2)
print(f"{len(classes)} classes, device={device}")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class RandomHRoll:
    """ERP-valid augmentation: a horizontal roll is a longitude rotation of the sphere."""
    def __call__(self, t):
        return torch.roll(t, shifts=random.randint(0, t.shape[2] - 1), dims=2)

train_tf = T.Compose([
    T.Resize((args.img_h, args.img_w)),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    RandomHRoll(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.Resize((args.img_h, args.img_w)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class ERPCls(Dataset):
    def __init__(self, root, stems, transform):
        self.root, self.stems, self.tf = root, stems, transform
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(os.path.join(self.root, "image", stem + ".png")).convert("RGB")
        return self.tf(img), cls2idx[stem.split("/")[0]]

def loader(split, tf, shuffle):
    ds = ERPCls(args.root, manifest[split], tf)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=(device == "cuda"), drop_last=shuffle)

train_dl = loader("train", train_tf, True)
val_dl = loader("val", eval_tf, False)
test_dl = loader("test", eval_tf, False)

def build_model():
    w = None if args.no_pretrained else "DEFAULT"
    m = getattr(torchvision.models, args.arch)(weights=w)
    n = len(classes)
    if args.arch in ("resnet18", "resnet50"):
        m.fc = nn.Linear(m.fc.in_features, n)
    elif args.arch == "efficientnet_b0":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif args.arch == "mobilenet_v3_large":
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif args.arch == "convnext_tiny":
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif args.arch == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, n)
    elif args.arch == "swin_t":
        m.head = nn.Linear(m.head.in_features, n)
    return m

model = build_model().to(device)
params_m = sum(p.numel() for p in model.parameters()) / 1e6
try:
    from ptflops import get_model_complexity_info
    with torch.no_grad():
        macs, _ = get_model_complexity_info(model, (3, args.img_h, args.img_w),
                                            as_strings=False, print_per_layer_stat=False, verbose=False)
    flops_g = 2 * macs / 1e9
except Exception:
    flops_g = float("nan")
print(f"[bench] arch={args.arch} params={params_m:.1f}M flops={flops_g:.1f}G")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
crit = nn.CrossEntropyLoss()
use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

@torch.no_grad()
def evaluate(dl):
    model.eval()
    loss_sum = correct = total = 0
    preds_all, labels_all = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            out = model(x)
            loss = crit(out, y)
        loss_sum += loss.item() * x.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total += x.size(0)
        preds_all.append(p.cpu()); labels_all.append(y.cpu())
    return loss_sum / total, correct / total, torch.cat(preds_all), torch.cat(labels_all)

hist = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}
best_acc = 0.0
for ep in range(1, args.epochs + 1):
    model.train()
    t0 = time.time(); ls = corr = tot = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        ls += loss.item() * x.size(0); corr += (out.argmax(1) == y).sum().item(); tot += x.size(0)
    sched.step()
    tr_loss, tr_acc = ls / tot, corr / tot
    va_loss, va_acc, _, _ = evaluate(val_dl)
    for k, v in zip(hist, [tr_loss, tr_acc, va_loss, va_acc]):
        hist[k].append(v)
    print(f"epoch {ep:2d}/{args.epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  "
          f"val {va_loss:.3f}/{va_acc:.3f}  ({time.time()-t0:.0f}s)")
    if va_acc >= best_acc:
        best_acc = va_acc
        torch.save({"model": model.state_dict(), "classes": classes, "arch": args.arch}, os.path.join(args.out, "best.pt"))

# final test with best checkpoint
ckpt = torch.load(os.path.join(args.out, "best.pt"), map_location=device)
model.load_state_dict(ckpt["model"])
te_loss, te_acc, preds, labels = evaluate(test_dl)
print(f"\nTEST  loss {te_loss:.3f}  top-1 {te_acc:.4f}  (best val {best_acc:.4f})")

n = len(classes)
cm = np.zeros((n, n), int)
for p, t in zip(preds.numpy(), labels.numpy()):
    cm[t, p] += 1
print("\nper-class accuracy:")
for i, c in enumerate(classes):
    tot = cm[i].sum()
    print(f"  {c:30s} {cm[i, i]/tot:.3f}" if tot else f"  {c:30s}   n/a")

tag = args.tag or f"{args.arch}_{args.img_h}x{args.img_w}"
bench_header = ["tag", "arch", "img_h", "img_w", "epochs", "params_m", "flops_g", "test_top1"] + classes
bench_row = [tag, args.arch, args.img_h, args.img_w, args.epochs,
             f"{params_m:.4f}", f"{flops_g:.4f}" if not np.isnan(flops_g) else "",
             f"{te_acc:.4f}"]
bench_row += [f"{cm[i, i] / cm[i].sum():.4f}" if cm[i].sum() else "" for i in range(n)]
bench_new = not os.path.exists(args.bench_csv)
with open(args.bench_csv, "a", newline="") as f:
    w = csv.writer(f)
    if bench_new:
        w.writerow(bench_header)
    w.writerow(bench_row)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif"})
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["tr_loss"], label="train"); ax[0].plot(ep, hist["va_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["tr_acc"], label="train"); ax[1].plot(ep, hist["va_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "curves.png"), dpi=130); plt.close(fig)

    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (row-normalized)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=130); plt.close(fig)
    print(f"\nsaved curves.png and confusion_matrix.png in {args.out}")
except Exception as e:
    print(f"[warn] figures skipped: {e}")
