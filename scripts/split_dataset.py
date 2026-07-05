#!/usr/bin/env python3
"""
Create a stratified train/val/test split manifest for the space ERP dataset.

Usage:
    python3 split_dataset.py /home/arbi/infinigen/outputs/space_erp_final
    python3 split_dataset.py <root> --val 0.1 --test 0.1 --seed 42

Splits *per class* (stratified), so every split stays class-balanced. Writes splits.json
containing relative stems like "01_earth_obs_satellite/00007" — from each stem the training
code locates image/<stem>.png, mask/<stem>.png, depth/<stem>.exr, meta/<stem>.json.
The split is deterministic given --seed, so it's reproducible across all tasks.
"""
import os, json, random, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--val", type=float, default=0.1)
ap.add_argument("--test", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=None)
a = ap.parse_args()

root = a.root.rstrip("/")
img = os.path.join(root, "image")
out = a.out or os.path.join(root, "splits.json")
rng = random.Random(a.seed)

per_class = defaultdict(list)
for cls in sorted(os.listdir(img)):
    d = os.path.join(img, cls)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if f.endswith(".png"):
            per_class[cls].append(f"{cls}/{os.path.splitext(f)[0]}")

splits = {"train": [], "val": [], "test": []}
report = {}
for cls, items in per_class.items():
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_test = int(round(n * a.test))
    n_val = int(round(n * a.val))
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    splits["test"] += test
    splits["val"] += val
    splits["train"] += train
    report[cls] = (len(train), len(val), len(test))

for k in splits:
    rng.shuffle(splits[k])

meta = {"seed": a.seed, "val_frac": a.val, "test_frac": a.test,
        "counts": {k: len(v) for k, v in splits.items()}, "splits": splits}
json.dump(meta, open(out, "w"))

print(f"wrote {out}")
tot = sum(len(v) for v in splits.values())
print(f"total {tot}  |  train {len(splits['train'])}  val {len(splits['val'])}  test {len(splits['test'])}")
print("per class  (train / val / test):")
for c in sorted(report):
    t, v, te = report[c]
    print(f"  {c:30s} {t:6d} / {v:5d} / {te:5d}")
