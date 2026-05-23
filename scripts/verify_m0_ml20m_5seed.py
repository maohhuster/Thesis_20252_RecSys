#!/usr/bin/env python3
"""Re-evaluate all 5 M0 (BPR-MF, ID-only) ML-20M checkpoints on the test set
and compare per-seed NDCG@10/Recall@10/MRR against the xlsx source-of-truth.

Diagnostic goal: determine whether seed-2026's `best_model.pt` (sha256
cc5418e3…, unchanged) is the genuine seed-2026 run or a mislabelled copy of
the seed-456 run. If seed-2026's freshly-evaluated metrics match the xlsx
seed-456 row instead of the seed-2026 row, the checkpoint is wrong.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "code" / "benchmark"
sys.path.insert(0, str(BENCH))

import torch
from config import EMBED_DIM
from data.dataset import InteractionData
from models.bpr_mf import BPRMF
from evaluate import evaluate_model

DATA_DIR = BENCH / "data" / "processed"
SEEDS = [42, 123, 456, 789, 2026]
CKPT = lambda s: BENCH / "checkpoints" / "m0" / "bge-large-en-v1.5" / f"seed-{s}" / "best_model.pt"

# xlsx 'bge-large-en-v1.5' sheet, M0 column, per-seed test metrics
XLSX = {
    42:   {"NDCG@10": 0.1136, "Recall@10": 0.0483, "MRR": 0.2391},
    123:  {"NDCG@10": 0.1134, "Recall@10": 0.0478, "MRR": 0.2344},
    456:  {"NDCG@10": 0.1135, "Recall@10": 0.0492, "MRR": 0.2311},
    789:  {"NDCG@10": 0.1135, "Recall@10": 0.0477, "MRR": 0.2360},
    2026: {"NDCG@10": 0.1143, "Recall@10": 0.0485, "MRR": 0.2360},
}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[verify-m0] device={device}")
    data = InteractionData(data_dir=DATA_DIR)
    print(f"[verify-m0] n_users={data.n_users}, n_items={data.n_items}")

    rows = {}
    for s in SEEDS:
        ck = CKPT(s)
        if not ck.exists():
            print(f"  seed-{s}: MISSING {ck}"); continue
        import hashlib
        h = hashlib.sha256(ck.read_bytes()).hexdigest()[:12]
        model = BPRMF(data.n_users, data.n_items, EMBED_DIM).to(device)
        sd = torch.load(ck, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        t0 = time.time()
        test = evaluate_model(model, data, split="test", device=device)
        dt = time.time() - t0
        rows[s] = {"sha12": h,
                   "test": {k: float(test[k]) for k in ("NDCG@10", "Recall@10", "MRR") if k in test},
                   "wall_s": round(dt, 1)}
        print(f"  seed-{s} (sha {h}) evaluated in {dt:.0f}s")

    print("\n" + "=" * 78)
    print("M0 ML-20M — per-seed test NDCG@10 vs xlsx (source of truth)")
    print("=" * 78)
    print(f"{'seed':>6} {'sha12':>13} {'eval NDCG':>10} {'xlsx NDCG':>10} {'Δ':>9}  verdict")
    for s in SEEDS:
        if s not in rows: continue
        ev = rows[s]["test"]["NDCG@10"]; xl = XLSX[s]["NDCG@10"]; d = ev - xl
        # which xlsx seed does this eval best match?
        best = min(SEEDS, key=lambda t: abs(ev - XLSX[t]["NDCG@10"]))
        verdict = "OK" if best == s and abs(d) <= 0.0010 else f"!! matches xlsx seed-{best}"
        print(f"{s:>6} {rows[s]['sha12']:>13} {ev:>10.4f} {xl:>10.4f} {d:>+9.4f}  {verdict}")

    out = REPO / "code" / "benchmark" / "results" / "verify_m0_ml20m_5seed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": device, "xlsx": XLSX, "evaluated": rows}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
