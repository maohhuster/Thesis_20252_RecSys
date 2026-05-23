"""Regenerate the combined sensitivity figure used in §6.4 of the paper.

Left panel: NDCG@10 for encoder-dependent configs under bge-large-en-v1.5 vs
e5-large-v2. Values are 5-seed means sourced from the xlsx workbook.

Right panel: Pearson r between Claude Haiku 4.5 and GPT-4o-mini mood vectors
across the 10 mood axes, sourced from human_eval/llm_comparison_results.json.

Output: paper/sensitivity_combined.pdf (+ docs/figures/.pdf/.png mirrors).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PAPER = ROOT / "paper" / "sensitivity_combined.pdf"
OUT_DOCS_PDF = ROOT / "docs" / "figures" / "sensitivity_combined.pdf"
OUT_DOCS_PNG = ROOT / "docs" / "figures" / "sensitivity_combined.png"

# ── Left panel data: 5-seed NDCG@10 means (paper-canonical, bge-large-en-v1.5)
configs = ["M3\n+BERT Title", "M4\n+LLM Profile", "M7\n+Profile+Mood", "M8\n+All LLM"]
bge_vals = [0.1135, 0.1177, 0.1184, 0.1135]
e5_vals = [0.1144, 0.1154, 0.1173, 0.1140]

# ── Right panel data: load from canonical JSON
with open(ROOT / "human_eval" / "llm_comparison_results.json") as f:
    comp = json.load(f)

axis_order = [
    "dark_light", "serious_playful", "realistic_fantastical", "slow_fast",
    "predictable_subversive", "intimate_epic", "cerebral_visceral",
    "emotional_detached", "conventional_experimental", "nostalgic_contemporary",
]
axis_labels = [
    "dark_light", "serious_playful", "realistic_fant.", "slow_fast",
    "predictable_subv.", "intimate_epic", "cerebral_visc.",
    "emotional_det.", "conv_exp.", "nostalgic_cont.",
]
mood_r = [comp["mood_correlation"][a]["pearson_r"] for a in axis_order]

# Color thresholds: green ≥ 0.70 (strong), yellow ≥ 0.40 (moderate), red < 0.40
def mood_color(r: float) -> str:
    if r >= 0.70:
        return "#2ca02c"
    if r >= 0.40:
        return "#ffb000"
    return "#d62728"

mood_colors = [mood_color(r) for r in mood_r]

# ── Plot
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

# Left: grouped bar chart
x = np.arange(len(configs))
width = 0.38
bars_bge = ax_l.bar(x - width/2, bge_vals, width, label="bge-large-en-v1.5",
                    color="#1f77b4")
bars_e5 = ax_l.bar(x + width/2, e5_vals, width, label="e5-large-v2",
                   color="#ff7f0e", hatch="//", edgecolor="white", linewidth=0)
for rect, val in zip(bars_bge, bge_vals):
    ax_l.text(rect.get_x() + rect.get_width()/2, val + 0.0002, f"{val:.4f}",
              ha="center", va="bottom", fontsize=8)
for rect, val in zip(bars_e5, e5_vals):
    ax_l.text(rect.get_x() + rect.get_width()/2, val + 0.0002, f"{val:.4f}",
              ha="center", va="bottom", fontsize=8)
ax_l.set_xticks(x)
ax_l.set_xticklabels(configs, fontsize=9)
ax_l.set_ylabel("NDCG@10")
ax_l.set_ylim(0.1050, 0.1250)
ax_l.set_title("Encoder Sensitivity\n(bge-large-en-v1.5 vs e5-large-v2, 5-seed mean)",
               fontsize=10)
ax_l.legend(loc="upper right", fontsize=8)
ax_l.grid(axis="y", linestyle="--", alpha=0.4)
ax_l.set_axisbelow(True)

# Right: horizontal bar chart of Pearson r
y_pos = np.arange(len(axis_labels))
ax_r.barh(y_pos, mood_r, color=mood_colors, edgecolor="none")
for i, (r, c) in enumerate(zip(mood_r, mood_colors)):
    x_text = r + (0.02 if r >= 0 else -0.02)
    ha = "left" if r >= 0 else "right"
    ax_r.text(x_text, i, f"{r:.2f}", va="center", ha=ha, fontsize=9, fontweight="bold")
ax_r.set_yticks(y_pos)
ax_r.set_yticklabels(axis_labels, fontsize=9)
ax_r.invert_yaxis()
ax_r.axvline(0, color="black", linewidth=0.8)
ax_r.set_xlim(-0.25, 1.05)
ax_r.set_xlabel("Pearson r (Claude Haiku 4.5 vs GPT-4o-mini)")
ax_r.set_title("LLM Sensitivity: Mood Vector Correlation\n(500 profiles, identical prompt)",
               fontsize=10)
ax_r.grid(axis="x", linestyle="--", alpha=0.4)
ax_r.set_axisbelow(True)

plt.tight_layout()

OUT_PAPER.parent.mkdir(parents=True, exist_ok=True)
OUT_DOCS_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PAPER, bbox_inches="tight")
fig.savefig(OUT_DOCS_PDF, bbox_inches="tight")
fig.savefig(OUT_DOCS_PNG, bbox_inches="tight", dpi=200)
print(f"Wrote {OUT_PAPER}")
print(f"Wrote {OUT_DOCS_PDF}")
print(f"Wrote {OUT_DOCS_PNG}")

# Sanity-check axis counts printed for the caption
green = sum(1 for r in mood_r if r >= 0.70)
yellow = sum(1 for r in mood_r if 0.40 <= r < 0.70)
red = sum(1 for r in mood_r if r < 0.40)
print(f"Axis color counts: green={green}, yellow={yellow}, red={red}; green+yellow={green+yellow}")
