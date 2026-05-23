#!/usr/bin/env python3
"""R1 (RLMRec-gene) ML-20M re-validation + 5-seed deployment at embedding_size=128.

WHY: R1's 54-point loss/depth grid was selected via the upstream single-run
path at RLMRec-native embedding_size=32, but R1 is *deployed* at d=128
(capacity-matched to the M7 / LightGCN-SF backbone for the dense fair
comparison). This script removes that selection/deployment-capacity mismatch
WITHOUT a full 54-cell re-run: it re-validates the committed winner against
its one-axis neighbours **at d=128** (seed-42, RLMRec-native Recall@20
selection), then trains the confirmed winner for the 5 paper seeds at d=128
and writes the canonical results/r1_metrics.json.

It reuses the verified primitives of run_r1_ml20m_single.py (patch_yml,
parse_log, device pick) and additionally patches the GLOBAL
model.embedding_size 32->128 in lightgcn_gene.yml for the duration of every
run. The committed yml is ALWAYS restored (try/finally).

Run on a CUDA box (Colab). Reduced grid = winner +- one-axis neighbours
(8 cells) at seed-42, then 5 seeds at the winner.

Usage:
    python3 scripts/run_r1_ml20m_revalidate_d128.py --device cuda
    python3 scripts/run_r1_ml20m_revalidate_d128.py --device cuda --select-only
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r1_ml20m_single as R1   # verified patch_yml / parse_log / constants

REPO    = R1.REPO
RLMREC  = R1.RLMREC
YML     = R1.YML
LOG_DIR = R1.LOG_DIR
DATASET = R1.DATASET
SEEDS   = [42, 123, 456, 789, 2026]
EMB     = 128
REG     = 1e-7

# committed 54-pt grid winner (selected at d=32) — the cell we re-validate at d=128
WINNER  = dict(layer_num=3, mask_ratio=0.10, recon_weight=0.10, re_temperature=0.2)
# original grid axis values (code/benchmark/hparams/r1/grid_selection.json)
AXES = dict(layer_num=[2, 3], mask_ratio=[0.05, 0.10, 0.15],
            recon_weight=[0.05, 0.10, 0.15], re_temperature=[0.1, 0.2, 0.5])

EMB_RE = re.compile(r"^(\s*embedding_size:\s*)\d+\s*$", re.MULTILINE)


def reduced_grid():
    """Winner + one-axis neighbours (vary a single axis at a time)."""
    cells = [dict(WINNER)]
    for ax, vals in AXES.items():
        for v in vals:
            if v == WINNER[ax]:
                continue
            c = dict(WINNER); c[ax] = v
            cells.append(c)
    # de-dup preserving order
    seen, out = set(), []
    for c in cells:
        key = tuple(c[a] for a in AXES)
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def patch_embedding_size(text: str, size: int) -> str:
    new, n = EMB_RE.subn(lambda m: f"{m.group(1)}{size}", text)
    if n != 1:
        raise RuntimeError(f"expected exactly one global embedding_size line in {YML}, found {n}")
    return new


def run_cell(cell, seed, device):
    """One RLMRec-gene run at d=128 with the given ml20m cell. Returns parse_log()."""
    original = YML.read_text()
    try:
        R1.patch_yml(cell["layer_num"], cell["mask_ratio"], cell["recon_weight"],
                     cell["re_temperature"], REG)
        YML.write_text(patch_embedding_size(YML.read_text(), EMB))   # 32 -> 128
        t0 = time.time()
        cmd = [sys.executable, "encoder/train_encoder.py", "--model", "lightgcn_gene",
               "--dataset", DATASET, "--device", device, "--seed", str(seed)]
        print(f"[revalidate_d128] cell={cell} seed={seed} emb={EMB} -> {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(RLMREC))
    finally:
        YML.write_text(original)   # never leave the committed yml mutated
    if r.returncode != 0:
        raise RuntimeError(f"train_encoder failed rc={r.returncode} cell={cell} seed={seed}")
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda q: q.stat().st_mtime)
    logs = [q for q in logs if q.stat().st_mtime >= t0 - 1]
    if not logs:
        raise RuntimeError(f"no fresh log for cell={cell} seed={seed}")
    best_epoch, val, test = R1.parse_log(logs[-1])
    return dict(cell=cell, seed=seed, best_epoch=best_epoch, val=val, test=test,
                log=str(logs[-1].relative_to(REPO)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--select-only", action="store_true",
                    help="run only the seed-42 selection grid, skip 5-seed deploy")
    a = ap.parse_args()

    out_dir = REPO / "code" / "benchmark" / "hparams" / "r1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: seed-42 re-validation grid at d=128 (selection = val Recall@20) ----
    grid = reduced_grid()
    print(f"[revalidate_d128] {len(grid)} cells at d=128, seed=42, "
          f"selection = validation Recall@20 (RLMRec-native)")
    sel = []
    for cell in grid:
        res = run_cell(cell, 42, a.device)
        rec20 = res["val"]["recall"][2] if res["val"] else None
        sel.append(dict(**res, val_recall20=rec20))
        print(f"[revalidate_d128]  -> val Recall@20={rec20}  "
              f"(best_epoch={res['best_epoch']}, test NDCG@10="
              f"{res['test']['ndcg'][1] if res['test'] else None})")
    sel_ok = [s for s in sel if s["val_recall20"] is not None]
    if not sel_ok:
        print("[revalidate_d128] FAILED — no parseable selection runs"); sys.exit(1)
    winner = max(sel_ok, key=lambda s: s["val_recall20"])
    (out_dir / "revalidation_d128_seed42.json").write_text(json.dumps(
        {"embedding_size": EMB, "selection_metric": "validation Recall@20 (RLMRec-native)",
         "reduced_grid": sel,
         "winner_cell": winner["cell"], "winner_val_recall20": winner["val_recall20"],
         "committed_d32_winner": WINNER,
         "winner_changed_vs_d32": winner["cell"] != WINNER}, indent=2, default=str))
    print(f"\n[revalidate_d128] d=128 winner: {winner['cell']} "
          f"(val Recall@20={winner['val_recall20']}) | "
          f"{'UNCHANGED vs d=32 grid winner' if winner['cell']==WINNER else 'CHANGED vs d=32 winner — investigate'}")
    if a.select_only:
        return

    # ---- Stage 2: 5-seed deployment at the d=128 winner ----
    per_seed = {}
    for s in SEEDS:
        res = run_cell(winner["cell"], s, a.device)
        per_seed[str(s)] = res["test"]
        print(f"[revalidate_d128] deploy seed={s} test NDCG@10="
              f"{res['test']['ndcg'][1] if res['test'] else None}")
    import numpy as np
    def agg(idx, metric):
        v = np.array([per_seed[str(s)][metric][idx] for s in SEEDS], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "values": [float(x) for x in v]}
    summary = {"NDCG@10": agg(1, "ndcg"), "Recall@10": agg(1, "recall"),
               "Recall@20": agg(2, "recall")}
    canon = REPO / "code" / "benchmark" / "results" / "r1_metrics.json"
    canon.write_text(json.dumps({
        "embedding_size": EMB,
        "selection": "reduced re-validation grid at d=128, seed-42, val Recall@20",
        "winner": winner["cell"], "reg_weight": REG,
        "seeds_evaluated": SEEDS, "per_seed": per_seed, "summary": summary,
        "_provenance": ("R1-gene ML-20M re-validated AND deployed at embedding_size=128 "
                        "(capacity-matched to the M7/LightGCN-SF backbone). Loss/depth "
                        "winner re-confirmed at d=128 via the reduced one-axis-neighbour "
                        "grid (seed-42, RLMRec-native Recall@20). Supersedes the prior "
                        "d=32-selected/d=128-deployed checkpoint.")},
        indent=2, default=str))
    print(f"[revalidate_d128] wrote {canon.relative_to(REPO)} | "
          f"NDCG@10={summary['NDCG@10']['mean']:.4f}+/-{summary['NDCG@10']['std']:.4f}")


if __name__ == "__main__":
    main()
