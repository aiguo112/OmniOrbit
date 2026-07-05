# Scripts

Run from the **repository root** (parent of this folder).

| Script | Description |
|---|---|
| `split_dataset.py` | Stratified train/val/test split → `splits.json` |
| `train_classify.py` | 13-class ResNet/EfficientNet classification baseline |
| `train_segment.py` | 14-class U-Net semantic segmentation baseline |
| `train_depth.py` | Scale-invariant monocular depth (SILog loss) |
| `train_pose.py` | 6-DoF attitude regression baseline |
| `subtypes.py` | Count frames per underlying 3-D model |
| `gather_gt.py` | Copy GT bundles for hand-picked candidate frames |
| `classify_candidates.py` | Classify `candidates_gt/` frames |
| `classify_testset.py` | Evaluate classifier on full test split |
| `evaluate_candidates.py` | Segmentation + depth eval on candidates |
| `make_all_figures.py` | Dataset stat charts from `stats.json` |
| `make_paper_figures.py` | Publication figures from checkpoints + test split |

See the **How to use this code** section in the root `README.md` for setup and example commands.
