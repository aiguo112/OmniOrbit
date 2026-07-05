#!/usr/bin/env python3
"""Regenerate README.md from stats.json and instructions/*.md templates."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTRUCTIONS = ROOT / "instructions"
STATS_PATH = ROOT / "stats.json"
OUTPUT_PATH = ROOT / "README.md"

# Semantic label palette (class name -> RGB, hex)
CLASS_PALETTE: dict[str, tuple[tuple[int, int, int], str]] = {
    "earth_obs_satellite": ((220, 50, 50), "#dc3232"),
    "space_telescope": ((50, 200, 80), "#32c850"),
    "space_station": ((50, 90, 220), "#325adc"),
    "deep_space_probe": ((220, 200, 50), "#dcc832"),
    "mars_mission": ((200, 100, 30), "#c8641e"),
    "comm_satellite": ((160, 60, 200), "#a03cc8"),
    "cubesat": ((60, 200, 200), "#3cc8c8"),
    "astronaut_eva": ((240, 130, 200), "#f082c8"),
    "solar_heliophysics": ((250, 160, 40), "#faa028"),
    "rocket_launch_vehicle": ((120, 80, 220), "#7850dc"),
    "planet": ((30, 120, 255), "#1e78ff"),
    "moon": ((190, 190, 190), "#bebebe"),
    "asteroid": ((120, 100, 80), "#786450"),
}

FIGURES: list[tuple[str, str]] = [
    (
        "Subject frames per class. A flat profile confirms class balance.",
        "figures/class_balance.png",
    ),
    (
        "Total occurrences per class. Planet/moon exceed their subject counts "
        "because they also appear as backdrops — relevant for per-pixel class weighting.",
        "figures/occurrences.png",
    ),
    (
        "Distinct underlying 3-D models per class. Low values (e.g. comm_satellite) "
        "indicate limited shape variety.",
        "figures/models_per_class.png",
    ),
    ("Share of frames carrying each attribute flag.", "figures/flags.png"),
    (
        "Number of labeled objects per frame (1 = subject only, 2 = subject + backdrop).",
        "figures/n_objects.png",
    ),
    (
        "Apparent angular size of the subject across the dataset.",
        "figures/angular_size.png",
    ),
    (
        "Per-class apparent size. Celestial classes appear larger than craft.",
        "figures/angular_size_by_class.png",
    ),
    ("Distance from camera to subject.", "figures/distance.png"),
    (
        "Subject azimuth distribution (flat = even 360° coverage).",
        "figures/azimuth.png",
    ),
    (
        "Subject elevation distribution; the tails are the deliberate near-pole frames.",
        "figures/elevation.png",
    ),
    (
        "Joint azimuth–elevation density. Uniform fill = good spherical coverage.",
        "figures/sky_coverage.png",
    ),
    (
        "Which planet/moon textures appear as backdrops.",
        "figures/backdrop_bodies.png",
    ),
    ("Per-model frame counts within planet.", "figures/subtypes_planet.png"),
    ("Per-model frame counts within moon.", "figures/subtypes_moon.png"),
    ("Per-model frame counts within asteroid.", "figures/subtypes_asteroid.png"),
]

INSTRUCTION_ORDER = [
    "_header.md",
    "how_to_use.md",
    "how_it_was_made.md",
    "assets_attribution.md",
    "comparison.md",
    "baseline_intro.md",
    "baseline_classification.md",
    "baseline_segmentation.md",
    "baseline_depth.md",
    "baseline_pose.md",
    "limitations.md",
]


def load_text(name: str) -> str:
    path = INSTRUCTIONS / name
    return path.read_text(encoding="utf-8").strip()


def load_stats() -> dict:
    with STATS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def section_at_a_glance(stats: dict) -> str:
    total = stats["total_images"]
    depth_ok = stats["flags"]["depth_ok"]
    models_total = sum(stats["models_per_class"].values())
    width, height = stats["resolution"]
    gpu = "True" if stats.get("gpu") else "False"

    lines = [
        "## At a glance",
        "",
        f"- **Total images:** {total:,}",
        f"- **Classes:** {stats['classes']}",
        f"- **Resolution:** {width}×{height}",
        f"- **Renderer:** {stats['engine']} ({gpu}), Blender {stats['blender']}",
        f"- **Distinct models:** {models_total} across {stats['classes']} classes",
        f"- **Depth coverage:** {depth_ok:,}/{total:,}",
    ]
    return "\n".join(lines)


def section_classes(stats: dict) -> str:
    lines = [
        "## Classes & label palette",
        "",
        "| Class | Subject frames | Models | RGB | Hex |",
        "|---|--:|--:|---|---|",
    ]
    for class_name, count in stats["subject_count"].items():
        models = stats["models_per_class"][class_name]
        rgb, hex_code = CLASS_PALETTE[class_name]
        rgb_str = f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
        lines.append(f"| {class_name} | {count:,} | {models} | {rgb_str} | {hex_code} |")

    lines.extend(["", "_Background (class 0) is black (0,0,0)._"])
    return "\n".join(lines)


def section_distribution(stats: dict) -> str:
    total = stats["total_images"]
    flags = stats["flags"]

    earth_backdrop = flags.get("earth_backdrop", 0)
    pole_case = flags.get("pole_case", 0)
    multi_object = flags.get("multi_object", 0)

    lines = [
        "## Distribution & balance",
        "",
        f"- Earth/planet backdrop: {fmt_pct(earth_backdrop, total)}",
        f"- Near-pole: {fmt_pct(pole_case, total)}",
        "- Pure-black background: 0.0%",
        f"- >1 labeled object: {fmt_pct(multi_object, total)}",
    ]
    return "\n".join(lines)


def section_figures() -> str:
    lines = ["## Figures", ""]
    for caption, path in FIGURES:
        lines.append(f"**{caption}**")
        lines.append("")
        lines.append(f"![{caption}]({path})")
        lines.append("")
    return "\n".join(lines).rstrip()


def generated_banner() -> str:
    today = date.today().isoformat()
    return (
        f"> Auto-generated by `make_dataset_docs.py` on {today}. "
        "Re-run after any re-render."
    )


def build_readme() -> str:
    stats = load_stats()
    parts = [
        load_text("_header.md"),
        "",
        generated_banner(),
        "",
        section_at_a_glance(stats),
        "",
        load_text("how_to_use.md"),
        "",
        load_text("how_it_was_made.md"),
        "",
        load_text("assets_attribution.md"),
        "",
        section_classes(stats),
        "",
        section_distribution(stats),
        "",
        section_figures(),
        "",
        load_text("comparison.md"),
        "",
        load_text("baseline_intro.md"),
        "",
        load_text("baseline_classification.md"),
        "",
        load_text("baseline_segmentation.md"),
        "",
        load_text("baseline_depth.md"),
        "",
        load_text("baseline_pose.md"),
        "",
        load_text("limitations.md"),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    readme = build_readme()
    OUTPUT_PATH.write_text(readme, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
