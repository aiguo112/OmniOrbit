#!/usr/bin/env python3
"""OmniOrbit candidate figures -> figures_candidates/ (vector PDF + PNG preview)."""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures_candidates"
os.makedirs(OUT, exist_ok=True)
STATS = str(Path(__file__).resolve().parent.parent / "stats.json")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 18, "axes.labelweight": "bold",
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "figure.dpi": 150,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name)

def pretty(s):
    return s.replace("_", " ")

def hbar(labels, vals, xlabel, name, color):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    labels = [pretty(labels[i]) for i in order]
    vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(labels, vals, color=color, edgecolor="black", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.14)
    for i, v in enumerate(vals):
        ax.text(v, i, "  " + format(v, ","), va="center", fontsize=14)
    fig.tight_layout()
    save(fig, name)

with open(STATS) as f:
    stats = json.load(f)

# --- Fig 1: subject distribution (balanced) ---
d = stats["subject_count"]
hbar(list(d), list(d.values()), "Subject frames", "01_subject_distribution", "#3b6ea5")

# --- Fig 2: occurrence distribution (subject + backdrop) ---
d = stats["occurrences"]
hbar(list(d), list(d.values()), "Total occurrences", "02_occurrence_distribution", "#9a6fb0")

# --- Fig 3: distinct models per class ---
d = stats["models_per_class"]
hbar(list(d), list(d.values()), "Distinct 3-D models", "03_models_per_class", "#5a9367")

print("done ->", OUT)
