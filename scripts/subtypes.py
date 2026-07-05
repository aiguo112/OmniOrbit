#!/usr/bin/env python3
"""
Per-class sub-type breakdown for the space dataset.

Usage:
    python3 subtypes.py /home/arbi/infinigen/outputs/space_erp_final

For each of the 13 classes it lists the distinct 3D models that appear AS THE SUBJECT
and how many frames each one got. The model name lives in every frame's metadata at
objects[].asset -- it is recorded per frame but is NOT a training label (only the 13
classes are). Writes subtypes.json: {class_name: {model_name: frame_count}}.
"""
import os, sys, json
from collections import defaultdict, Counter

root = sys.argv[1] if len(sys.argv) > 1 else "/home/arbi/infinigen/outputs/space_erp_final"
meta_root = os.path.join(root, "meta")

by_class = defaultdict(Counter)          # class_name -> Counter(model -> frames as subject)
for sub in sorted(os.listdir(meta_root)):
    d = os.path.join(meta_root, sub)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if not f.endswith(".json"):
            continue
        try:
            m = json.load(open(os.path.join(d, f)))
        except Exception:
            continue
        cls = m.get("primary_class", "?")
        objs = m.get("objects", [])
        # the subject is the object whose class matches the frame's primary class
        subj = next((o for o in objs if o.get("class_name") == cls), objs[0] if objs else None)
        if subj:
            by_class[cls][subj.get("asset", "?")] += 1

out, grand, total_types = {}, 0, 0
for cls in sorted(by_class):
    models = by_class[cls]
    total = sum(models.values()); grand += total; total_types += len(models)
    print(f"\n=== {cls}  ({len(models)} sub-types, {total} frames) ===")
    for name, cnt in models.most_common():
        print(f"   {cnt:5d}  {name}")
    out[cls] = dict(models.most_common())

print(f"\nTOTAL subject frames        : {grand}")
print(f"TOTAL distinct sub-types     : {total_types}  (across {len(by_class)} classes)")
json.dump(out, open(os.path.join(root, "subtypes.json"), "w"), indent=2)
print(f"Wrote {os.path.join(root, 'subtypes.json')}  (class -> model -> frame count)")
