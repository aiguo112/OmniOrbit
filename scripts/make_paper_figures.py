#!/usr/bin/env python3
"""Publication-quality figures for the OmniOrbit paper / dataset docs.

Reads training logs (where available), re-evaluates saved checkpoints on the test
split for per-class metrics and the classification confusion matrix, and writes
polished PNG + PDF files to figures_pub/.

Run from outputs/:
    conda activate pano360x
    python3 make_paper_figures.py --root space_erp_final --ckpt runs/classify_384/best.pt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def short_name(full: str) -> str:
    return full.split("_", 1)[-1] if "_" in full else full


def parse_segment_log(path: Path):
    loss, miou, miou_fg = [], [], []
    for line in path.read_text().splitlines():
        m = re.search(
            r"epoch\s+\d+/\d+\s+loss\s+([\d.]+)\s+val mIoU\s+([\d.]+)\s+\(fg\s+([\d.]+)\)",
            line,
        )
        if m:
            loss.append(float(m.group(1)))
            miou.append(float(m.group(2)))
            miou_fg.append(float(m.group(3)))
    return loss, miou, miou_fg


def parse_depth_log(path: Path):
    loss, absrel, d1 = [], [], []
    for line in path.read_text().splitlines():
        m = re.search(
            r"epoch\s+\d+/\d+\s+loss\s+([\d.]+)\s+val AbsRel\s+([\d.]+)\s+RMSE\s+[\d.]+\s+d1\s+([\d.]+)",
            line,
        )
        if m:
            loss.append(float(m.group(1)))
            absrel.append(float(m.group(2)))
            d1.append(float(m.group(3)))
    return loss, absrel, d1


def parse_classify_log(path: Path):
    tr_loss, tr_acc, va_loss, va_acc = [], [], [], []
    for line in path.read_text().splitlines():
        m = re.search(
            r"epoch\s+\d+/\d+\s+train\s+([\d.]+)/([\d.]+)\s+val\s+([\d.]+)/([\d.]+)",
            line,
        )
        if m:
            tr_loss.append(float(m.group(1)))
            tr_acc.append(float(m.group(2)))
            va_loss.append(float(m.group(3)))
            va_acc.append(float(m.group(4)))
    return tr_loss, tr_acc, va_loss, va_acc


def parse_pose_log(path: Path):
    loss, median = [], []
    for line in path.read_text().splitlines():
        m = re.search(
            r"epoch\s+\d+/\d+\s+loss\s+([\d.]+)\s+val median\s+([\d.]+)deg",
            line,
        )
        if m:
            loss.append(float(m.group(1)))
            median.append(float(m.group(2)))
    return loss, median


def plot_cls_curves(out_dir, tr_loss, tr_acc, va_loss, va_acc):
    ep = range(1, len(tr_loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(ep, tr_loss, label="train", lw=1.5)
    ax[0].plot(ep, va_loss, label="val", lw=1.5)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Classification loss")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[1].plot(ep, tr_acc, label="train", lw=1.5)
    ax[1].plot(ep, va_acc, label="val", lw=1.5)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Top-1 accuracy")
    ax[1].set_title("Classification accuracy")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save_fig(fig, out_dir, "cls_curves")


def plot_seg_curves(out_dir, loss, miou, miou_fg):
    ep = range(1, len(loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(ep, loss, color="#333333", lw=1.5)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Train loss")
    ax[0].set_title("Segmentation loss")
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[1].plot(ep, miou, label="mIoU", lw=1.5)
    ax[1].plot(ep, miou_fg, label="foreground mIoU", lw=1.5)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("mIoU")
    ax[1].set_title("Validation mIoU")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save_fig(fig, out_dir, "seg_curves")


def plot_depth_curves(out_dir, loss, absrel, d1):
    ep = range(1, len(loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(ep, loss, color="#333333", lw=1.5)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("SILog loss")
    ax[0].set_title("Depth training loss")
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[1].plot(ep, absrel, label="AbsRel", lw=1.5)
    ax[1].plot(ep, d1, label=r"$\delta<1.25$", lw=1.5)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric value")
    ax[1].set_title("Validation depth metrics")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save_fig(fig, out_dir, "depth_curves")


def plot_pose_curves(out_dir, loss, median):
    ep = range(1, len(loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(ep, loss, color="#333333", lw=1.5)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Train loss")
    ax[0].set_title("Pose training loss")
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[1].plot(ep, median, color="#c86432", lw=1.5)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Median error (deg)")
    ax[1].set_title("Validation median geodesic error")
    ax[1].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save_fig(fig, out_dir, "pose_curves")


def republish_png(src: Path, out_dir: Path, stem: str):
    img = plt.imread(src)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    save_fig(fig, out_dir, stem)


def plot_confusion_matrix(classes, cm, out_dir):
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    labels = [short_name(c) for c in classes]
    n = len(classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Fraction", rotation=270, labelpad=12)
    fig.tight_layout()
    save_fig(fig, out_dir, "cls_confusion_matrix")


def plot_per_class_bar(out_dir, stem, names, values, xlabel, title, color="#3c78c8"):
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.barh(range(len(names)), values, color=color, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short_name(n) for n in names], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def eval_classify(root: Path, ckpt: Path, img_h=384, img_w=768):
    import torch
    import torch.nn as nn
    import torchvision
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    classes = ck["classes"]
    arch = ck.get("arch", "resnet50")
    manifest = __import__("json").load(open(root / "splits.json"))["splits"]
    cls2idx = {c: i for i, c in enumerate(classes)}
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    class DS(Dataset):
        def __init__(self, stems):
            self.stems = stems

        def __len__(self):
            return len(self.stems)

        def __getitem__(self, i):
            stem = self.stems[i]
            img = Image.open(root / "image" / f"{stem}.png").convert("RGB")
            img = img.resize((img_w, img_h))
            arr = np.asarray(img, np.float32) / 255.0
            arr = (arr - np.array(mean)) / np.array(std)
            return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float(), cls2idx[
                stem.split("/")[0]
            ]

    dl = DataLoader(DS(manifest["test"]), batch_size=32, shuffle=False, num_workers=4)

    if arch == "resnet50":
        model = torchvision.models.resnet50(weights=None)
    else:
        model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    n = len(classes)
    cm = np.zeros((n, n), int)
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                cm[t, p] += 1

    accs = []
    for i in range(n):
        tot = cm[i].sum()
        accs.append(cm[i, i] / tot if tot else 0.0)
    return classes, cm, accs


def eval_segment(root: Path, ckpt: Path, img_h=512, img_w=1024):
    import torch
    import segmentation_models_pytorch as smp
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    names = ck["names"]
    num = len(names)
    manifest = __import__("json").load(open(root / "splits.json"))["splits"]
    palette = {
        0: (0, 0, 0),
        1: (220, 50, 50),
        2: (50, 200, 80),
        3: (50, 90, 220),
        4: (220, 200, 50),
        5: (200, 100, 30),
        6: (160, 60, 200),
        7: (60, 200, 200),
        8: (240, 130, 200),
        9: (250, 160, 40),
        10: (120, 80, 220),
        11: (30, 120, 255),
        12: (190, 190, 190),
        13: (120, 100, 80),
    }
    p_arr = np.array([palette[i] for i in range(num)], np.int64)
    p_pack = (p_arr[:, 0] << 16) | (p_arr[:, 1] << 8) | p_arr[:, 2]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def mask_to_label(rgb):
        packed = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        label = np.zeros(packed.shape, np.uint8)
        for idx in range(num):
            label[packed == p_pack[idx]] = idx
        return label

    class DS(Dataset):
        def __init__(self, stems):
            self.stems = stems

        def __len__(self):
            return len(self.stems)

        def __getitem__(self, i):
            stem = self.stems[i]
            img = np.asarray(
                Image.open(root / "image" / f"{stem}.png").convert("RGB").resize((img_w, img_h))
            , np.float32)
            m = np.asarray(Image.open(root / "mask" / f"{stem}.png").convert("RGB"))
            lab = mask_to_label(m)
            lab = np.asarray(Image.fromarray(lab).resize((img_w, img_h), Image.NEAREST))
            img = (img / 255.0 - mean) / std
            return torch.from_numpy(img.transpose(2, 0, 1).copy()).float(), torch.from_numpy(
                lab.astype(np.int64)
            )

    dl = DataLoader(DS(manifest["test"]), batch_size=8, shuffle=False, num_workers=4)
    model = smp.Unet(
        encoder_name=ck.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=num,
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    cm = np.zeros((num, num), np.int64)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            k = y.reshape(-1) * num + pred.reshape(-1)
            cm += np.bincount(k, minlength=num * num).reshape(num, num)

    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    valid = union > 0
    # foreground classes only (exclude background index 0)
    fg_names = names[1:]
    fg_iou = [float(iou[i]) for i in range(1, num) if valid[i]]
    return fg_names, fg_iou


def eval_pose(root: Path, ckpt: Path, crop=224, pad=0.25):
    import json

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    manifest = json.load(open(root / "splits.json"))["splits"]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def meta_path(stem):
        return root / "meta" / f"{stem}.json"

    def load_pose(stem):
        m = json.load(open(meta_path(stem)))
        pc = m.get("primary_class")
        subj = None
        for o in m.get("objects", []):
            if o.get("class_id") == pc:
                subj = o
                break
        if subj is None and m.get("objects"):
            subj = m["objects"][0]
        q = np.array(subj["rotation_quat"], np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        az = np.deg2rad(float(subj.get("azimuth", 0.0)))
        el = np.deg2rad(float(subj.get("elevation", 0.0)))
        bearing = np.array([np.sin(az), np.cos(az), np.sin(el), np.cos(el)], np.float32)
        return q, bearing

    def object_crop(img, mask):
        nb = mask.max(axis=2) > 0
        if not nb.any():
            return img
        w = mask.shape[1]
        if nb[:, 0].any() and nb[:, -1].any():
            img = np.roll(img, w // 2, axis=1)
            mask = np.roll(mask, w // 2, axis=1)
            nb = mask.max(axis=2) > 0
        ys, xs = np.where(nb)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, wbox = y1 - y0 + 1, x1 - x0 + 1
        py, px = int(h * pad), int(wbox * pad)
        y0 = max(0, y0 - py)
        y1 = min(img.shape[0], y1 + py + 1)
        x0 = max(0, x0 - px)
        x1 = min(img.shape[1], x1 + px + 1)
        return img[y0:y1, x0:x1]

    class DS(Dataset):
        def __init__(self, stems):
            self.stems = stems

        def __len__(self):
            return len(self.stems)

        def __getitem__(self, i):
            stem = self.stems[i]
            img = np.asarray(Image.open(root / "image" / f"{stem}.png").convert("RGB"))
            mask = np.asarray(Image.open(root / "mask" / f"{stem}.png").convert("RGB"))
            crop_img = object_crop(img, mask)
            pil = Image.fromarray(crop_img).resize((crop, crop), Image.BILINEAR)
            arr = (np.asarray(pil, np.float32) / 255.0 - mean) / std
            q, bearing = load_pose(stem)
            return (
                torch.from_numpy(arr.transpose(2, 0, 1).copy()).float(),
                torch.from_numpy(bearing),
                torch.from_numpy(q),
                stem.split("/")[0],
            )

    dl = DataLoader(DS(manifest["test"]), batch_size=64, shuffle=False, num_workers=4)

    class PoseNet(nn.Module):
        def __init__(self, arch):
            super().__init__()
            bb = getattr(torchvision.models, arch)(weights=None)
            self.fdim = bb.fc.in_features
            bb.fc = nn.Identity()
            self.backbone = bb
            self.head = nn.Sequential(
                nn.Linear(self.fdim + 4, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 4),
            )

        def forward(self, x, bearing):
            f = self.backbone(x)
            return self.head(torch.cat([f, bearing], dim=1))

    model = PoseNet(ck.get("arch", "resnet50"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    errs, classes = [], []
    with torch.no_grad():
        for x, b, q, cls in dl:
            x, b, q = x.to(device), b.to(device), q.to(device)
            pred = F.normalize(model(x, b).float(), dim=1)
            dot = (pred * q).sum(1).abs().clamp(max=1.0)
            deg = torch.rad2deg(2 * torch.arccos(dot)).cpu().numpy()
            errs.extend(deg.tolist())
            classes.extend(list(cls))

    errs = np.array(errs)
    classes = np.array(classes)
    per = {}
    for c in sorted(set(classes)):
        per[c] = float(np.median(errs[classes == c]))
    names = list(per.keys())
    values = [per[n] for n in names]
    return names, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="space_erp_final")
    ap.add_argument("--ckpt", default="runs/classify_384/best.pt")
    ap.add_argument("--seg-ckpt", default="runs/segment_full/best.pt")
    ap.add_argument("--depth-ckpt", default="runs/depth_full/best.pt")
    ap.add_argument("--pose-ckpt", default="runs/pose_test2/best.pt")
    ap.add_argument("--out", default="figures_pub")
    ap.add_argument("--seg-log", default="runs/segment_full.log")
    ap.add_argument("--depth-log", default="runs/depth_full.log")
    ap.add_argument("--cls-log", default=None)
    ap.add_argument("--pose-log", default=None)
    args = ap.parse_args()

    setup_style()
    root = (HERE / args.root).resolve()
    out_dir = (HERE / args.out).resolve()
    ckpt = (HERE / args.ckpt).resolve()

    print(f"root={root}\nout={out_dir}")

    # --- curves from logs (or republish PNG fallback) ---
    seg_log = HERE / args.seg_log
    if seg_log.is_file():
        plot_seg_curves(out_dir, *parse_segment_log(seg_log))
    elif (HERE / "runs/segment_full/curves.png").is_file():
        republish_png(HERE / "runs/segment_full/curves.png", out_dir, "seg_curves")

    depth_log = HERE / args.depth_log
    if depth_log.is_file():
        plot_depth_curves(out_dir, *parse_depth_log(depth_log))
    elif (HERE / "runs/depth_full/curves.png").is_file():
        republish_png(HERE / "runs/depth_full/curves.png", out_dir, "depth_curves")

    cls_log = Path(args.cls_log) if args.cls_log else None
    if cls_log and cls_log.is_file():
        plot_cls_curves(out_dir, *parse_classify_log(cls_log))
    elif (ckpt.parent / "curves.png").is_file():
        republish_png(ckpt.parent / "curves.png", out_dir, "cls_curves")

    pose_log = Path(args.pose_log) if args.pose_log else None
    if pose_log and pose_log.is_file():
        plot_pose_curves(out_dir, *parse_pose_log(pose_log))
    elif (HERE / "runs/pose_test2/curves.png").is_file():
        republish_png(HERE / "runs/pose_test2/curves.png", out_dir, "pose_curves")

    # --- re-evaluate for confusion matrix + per-class bars ---
    print("\nRe-evaluating classification checkpoint for confusion matrix ...")
    try:
        classes, cm, accs = eval_classify(root, ckpt)
        plot_confusion_matrix(classes, cm, out_dir)
        plot_per_class_bar(
            out_dir,
            "cls_per_class",
            classes,
            accs,
            "Test top-1 accuracy",
            "Per-class classification accuracy",
            color="#325adc",
        )
    except Exception as e:
        print(f"[ERROR] classification confusion matrix / per-class failed: {type(e).__name__}: {e}")

    print("Re-evaluating segmentation checkpoint ...")
    try:
        seg_names, seg_iou = eval_segment(root, (HERE / args.seg_ckpt).resolve())
        plot_per_class_bar(
            out_dir,
            "seg_per_class",
            seg_names,
            seg_iou,
            "Test IoU",
            "Per-class segmentation IoU (foreground)",
            color="#32c850",
        )
    except Exception as e:
        print(f"[ERROR] segmentation per-class failed: {type(e).__name__}: {e}")

    print("Re-evaluating pose checkpoint ...")
    try:
        pose_names, pose_med = eval_pose(root, (HERE / args.pose_ckpt).resolve())
        plot_per_class_bar(
            out_dir,
            "pose_per_class",
            pose_names,
            pose_med,
            "Median geodesic error (deg)",
            "Per-class attitude error (median)",
            color="#c86432",
        )
    except Exception as e:
        print(f"[ERROR] pose per-class failed: {type(e).__name__}: {e}")

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
