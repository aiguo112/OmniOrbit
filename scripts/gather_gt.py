#!/usr/bin/env python3
"""Gather GT (mask/depth/meta) for each hand-picked candidate -> candidates_gt/<class>/<stem>/."""
import os, glob, shutil

ROOT = "space_erp_final"
CAND = os.path.join(ROOT, "image", "Best Candidates")
OUT  = "candidates_gt"
SRC = {"image": ("image", ".png"),
       "mask":  ("mask",  ".png"),
       "depth": ("depth", ".exr"),
       "meta":  ("meta",  ".json")}

frames = sorted(glob.glob(os.path.join(CAND, "*", "*.png")))
print(f"{len(frames)} candidate frames")

missing = []
for fp in frames:
    cls  = os.path.basename(os.path.dirname(fp))
    stem = os.path.splitext(os.path.basename(fp))[0]
    od = os.path.join(OUT, cls, stem)
    os.makedirs(od, exist_ok=True)
    for kind, (subdir, ext) in SRC.items():
        src = os.path.join(ROOT, subdir, cls, stem + ext)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(od, kind + ext))
        else:
            missing.append((cls, stem, kind, src))

print(f"bundled -> {OUT}/")
if missing:
    print(f"\n{len(missing)} MISSING ground-truth files:")
    for cls, stem, kind, src in missing[:40]:
        print(f"  [{cls}] {stem}: no {kind}  ({src})")
    if len(missing) > 40:
        print(f"  ... and {len(missing)-40} more")
else:
    print("all candidates have complete GT (image/mask/depth/meta)")
