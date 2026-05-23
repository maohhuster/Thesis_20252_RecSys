#!/usr/bin/env python3
"""Regenerate paper/density_chain.pdf with the canonical R1-gene/R1-plus
delta-vs-M7 chain across the 4 density datapoints.

The chain is the load-bearing visualization for the density-paradigm framework:
both regularizer instantiations (R1-gene = generative reconstruction; R1-plus =
contrastive distillation) must be monotone in 1/density. Numbers are pulled
from the canonical artifacts (hparams/r1/grid_selection.json for R1-gene
ML-20M at d=128; results files + paper appendix for the others).
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

# Density datapoints (int/item) — paper-canonical
DATAPOINTS = [
    ("ML-20M",       1160),
    (r"ML-20M$_{\mathrm{sub}}$",       225),
    ("ML-1M",         163),
    ("Amazon-Books",   13),
]

# R1-gene chain vs M7 (NDCG@10 percent delta)
# - ML-20M:        d=128 capacity-matched; 0.1162 vs M7 0.1175 -> -1.1% (p=0.007)
# - ML-20M_sub163: +3.7% (p=0.014; same-domain control)
# - ML-1M:        +17.0%
# - Amazon-Books: +29.5%
R1_GENE = [-1.1, +3.7, +17.0, +29.5]

# R1-plus chain vs M7
# - ML-20M:        d=128 capacity-matched; 0.1109 vs M7 0.1175 -> -5.6% (p=3e-5; all 5 seeds same-sign)
# - ML-20M_sub163: +10.9% (p<0.001; same-domain control)
# - ML-1M:        +16.3%
# - Amazon-Books: +26.4%
R1_PLUS = [-5.6, +10.9, +16.3, +26.4]

xs = [1.0 / d for d, _ in zip([d for _, d in DATAPOINTS], DATAPOINTS)]
# Above is silly; just:
xs = [1.0 / d for _, d in DATAPOINTS]

fig, ax = plt.subplots(figsize=(3.62, 2.89))
ax.plot(xs, R1_GENE, "o-", color="#1f77b4", lw=1.6, markersize=6, label="R1-gene (generative)")
ax.plot(xs, R1_PLUS, "s-", color="#d62728", lw=1.6, markersize=6, label="R1-plus (contrastive)")
ax.axhline(0, color="grey", lw=0.7, ls="--")

# Datapoint labels (annotate above each x)
label_offsets = [(0, -3.5), (-25, 6), (0, 4), (-65, 2.5)]  # (xpts, ypts) offsets
for (label, _), x, y, (dx, dy) in zip(DATAPOINTS, xs, R1_GENE, label_offsets):
    ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color="0.3", ha="left" if dx >= 0 else "left")

# Same-domain density-control callout
sub_x, sub_y = xs[1], R1_GENE[1]
ax.scatter([sub_x], [sub_y], s=160, facecolor="none", edgecolor="green", lw=1.3, zorder=5)
ax.annotate("same-domain\ndensity control",
            xy=(sub_x, sub_y), xytext=(45, -32), textcoords="offset points",
            fontsize=8, color="green",
            arrowprops=dict(arrowstyle="-", color="green", lw=0.7))

ax.set_xscale("log")
ax.set_xlabel(r"1 / density   (sparser $\to$)", fontsize=10)
ax.set_ylabel(r"$\Delta\%$ NDCG@10 vs. M7", fontsize=10)
ax.set_ylim(-12, 34)
ax.tick_params(labelsize=9)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
# Output paths: default to paper/ in this repo; CIKM mirror path is configurable
# via the LLM_MOVIELENS_CIKM_PAPER_DIR env var (set externally; not required).
import os
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
out = repo_root / "paper" / "density_chain.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"wrote {out}")

# Optional mirror to a sibling CIKM paper directory if configured.
cikm_dir = os.environ.get("LLM_MOVIELENS_CIKM_PAPER_DIR")
if cikm_dir:
    import shutil
    cikm_out = Path(cikm_dir) / "density_chain.pdf"
    shutil.copy2(out, cikm_out)
    print(f"wrote {cikm_out}")
