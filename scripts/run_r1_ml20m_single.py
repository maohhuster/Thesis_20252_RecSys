#!/usr/bin/env python3
"""Single R1 (RLMRec-gene) ML-20M run with explicitly-chosen hyperparameters.

Companion to scripts/run_r1_ml20m_grid.py: same patch/parse logic, but for ONE
arbitrary config instead of the 54-point sweep. Use it for:
  * spot-checking a single grid cell (e.g. layer_num=2 to see the depth effect),
  * the reg_weight=1e-6 reviewer control (the one param the grid fixes),
  * any one-off sensitivity probe.

It patches the `ml20m:` block of lightgcn_gene.yml, runs RLMRec's upstream
trainer (selection/early-stopping = best validation Recall@20, RLMRec's
native criterion — NOT NDCG@10), records the metrics at that selected best
epoch (val + test, all K), and ALWAYS restores the committed yml (try/finally). It does NOT touch
code/benchmark/hparams/r1/grid_selection/ unless you point --out there, so it can't
accidentally pollute the official grid.

Examples:
    # depth control: same as grid winner but layer_num=2
    python3 scripts/run_r1_ml20m_single.py \
        --layer_num 2 --mask_ratio 0.10 --recon_weight 0.10 --re_temperature 0.2 \
        --out code/benchmark/hparams/r1/probes/ln2_winner_seed42.json

    # reviewer reg_weight control (the param the grid fixes)
    python3 scripts/run_r1_ml20m_single.py \
        --layer_num 3 --mask_ratio 0.10 --recon_weight 0.10 --re_temperature 0.2 \
        --reg_weight 1e-6 --out code/benchmark/hparams/r1/probes/regw1e6_seed42.json
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

REPO    = Path(__file__).resolve().parent.parent
RLMREC  = REPO / "code" / "benchmark" / "external" / "RLMRec"
YML     = RLMREC / "encoder" / "config" / "modelconf" / "lightgcn_gene.yml"
LOG_DIR = RLMREC / "encoder" / "log" / "lightgcn_gene"

DATASET = "ml20m"   # data dir: code/benchmark/external/RLMRec/data/ml20m
ML20M_BLOCK_RE = re.compile(r"(^  ml20m:\n)(?:^    .*\n)+", re.MULTILINE)


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def patch_yml(ln, mr, rw, rt, reg):
    block = (
        "  ml20m:\n"
        f"    layer_num: {ln}\n"
        f"    reg_weight: {reg}\n"
        f"    mask_ratio: {mr}\n"
        f"    recon_weight: {rw}\n"
        f"    re_temperature: {rt}\n"
    )
    text = YML.read_text()
    new, n = ML20M_BLOCK_RE.subn(block, text)
    if n != 1:
        raise RuntimeError(f"could not locate a unique `ml20m:` block in {YML}")
    YML.write_text(new)


def _vec(line: str, metric: str):
    """Parse a RLMRec bracket line into [@5, @10, @20] for `metric` in {recall,ndcg}."""
    out = []
    for k in (5, 10, 20):
        m = re.search(rf"{metric}@{k}:\s*([0-9.]+)", line)
        out.append(float(m.group(1)) if m else None)
    return out


def parse_log(path: Path):
    """Extract the metrics at the SELECTED (best) epoch — the epoch whose
    checkpoint RLMRec actually saves — NOT the running max over epochs.

    RLMRec prints `Best Epoch <N>. Final test result: {...}` and a per-epoch
    `Epoch <N> Validation set [recall@5/10/20] [ndcg@5/10/20]` line. We tie the
    reported validation metrics to the best epoch N (the saved model), and read
    the test metrics from the authoritative `Final test result` arrays.

    Returns (best_epoch, val, test) where val/test are
    {"recall": [@5,@10,@20], "ndcg": [@5,@10,@20]} (or None if unparsable).
    """
    lines = path.read_text().splitlines()
    text = "\n".join(lines)

    m_be = re.search(r"Best Epoch\s+(\d+)", text)
    best_epoch = int(m_be.group(1)) if m_be else None

    val = None
    if best_epoch is not None:
        for ln in lines:
            if (re.search(rf"Epoch\s+0*{best_epoch}\s+Validation set", ln)
                    and "ndcg@10" in ln and "recall@10" in ln):
                val = {"recall": _vec(ln, "recall"), "ndcg": _vec(ln, "ndcg")}
                break
    # Fallback: the un-prefixed best-epoch validation summary line RLMRec prints
    # right before "Save model parameters" (first of the two trailing bracket lines).
    if val is None:
        for i, ln in enumerate(lines):
            if "Save model parameters" in ln:
                for back in (i - 2, i - 1):
                    if 0 <= back < len(lines) and "ndcg@10" in lines[back] \
                            and "Validation set" not in lines[back]:
                        val = {"recall": _vec(lines[back], "recall"),
                                "ndcg": _vec(lines[back], "ndcg")}
                        break
                break

    test = None
    m_t = re.search(
        r"Final test result:.*?'recall':\s*array\(\[([^\]]+)\].*?"
        r"'ndcg':\s*array\(\[([^\]]+)\]", text, re.S)
    if m_t:
        test = {"recall": [float(x) for x in m_t.group(1).split(",")],
                "ndcg":   [float(x) for x in m_t.group(2).split(",")]}
    return best_epoch, val, test


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer_num",      type=int,   required=True)
    p.add_argument("--mask_ratio",     type=float, required=True)
    p.add_argument("--recon_weight",   type=float, required=True)
    p.add_argument("--re_temperature", type=float, required=True)
    p.add_argument("--reg_weight",     type=float, default=1e-7,
                   help="default 1e-7 = upstream dense-dataset value (the grid "
                        "fixes this; override only for the reviewer control)")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    p.add_argument("--out",    required=True, help="JSON output path")
    a = p.parse_args()

    device = pick_device(a.device)
    out_path = Path(a.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tag = (f"ln{a.layer_num}_mr{a.mask_ratio}_rw{a.recon_weight}"
           f"_rt{a.re_temperature}_reg{a.reg_weight}_seed{a.seed}")
    print(f"[r1_single] {DATASET} {tag} device={device}")
    print(f"[r1_single] out={out_path}")

    original = YML.read_text()
    t0 = time.time()
    try:
        patch_yml(a.layer_num, a.mask_ratio, a.recon_weight,
                  a.re_temperature, a.reg_weight)
        cmd = [sys.executable, "encoder/train_encoder.py",
               "--model", "lightgcn_gene", "--dataset", DATASET,
               "--device", device, "--seed", str(a.seed)]
        print(f"[r1_single] → {' '.join(cmd)}  (cwd={RLMREC})")
        r = subprocess.run(cmd, cwd=str(RLMREC))
    finally:
        YML.write_text(original)   # never leave the committed yml mutated
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[r1_single] FAILED (rc={r.returncode}) after {dt/60:.1f} min")
        sys.exit(1)

    logs = sorted(LOG_DIR.glob("*.log"), key=lambda q: q.stat().st_mtime)
    logs = [q for q in logs if q.stat().st_mtime >= t0 - 1]
    if not logs:
        print("[r1_single] FAILED — no fresh log found")
        sys.exit(1)
    best_epoch, val, test = parse_log(logs[-1])
    if val is None or test is None:
        print(f"[r1_single] WARN — could not parse "
              f"{'val' if val is None else ''}{'/' if val is None and test is None else ''}"
              f"{'test' if test is None else ''} metrics from {logs[-1]}")

    def at(d, metric, k):
        if not d:
            return None
        idx = {5: 0, 10: 1, 20: 2}[k]
        v = d.get(metric)
        return v[idx] if v and len(v) > idx else None

    out = {
        "method": "R1 = RLMRec-gene (generative reconstruction regularizer)",
        "dataset": DATASET, "seed": a.seed, "device": device,
        "layer_num": a.layer_num, "mask_ratio": a.mask_ratio,
        "recon_weight": a.recon_weight, "re_temperature": a.re_temperature,
        "reg_weight": a.reg_weight,
        "best_epoch": best_epoch,
        "selection_metric": "Recall@20 (RLMRec upstream-native early-stopping)",
        # Metrics AT the selected (best) epoch — the epoch whose checkpoint is saved.
        "val": val,    # {"recall":[@5,@10,@20], "ndcg":[@5,@10,@20]} at best epoch
        "test": test,  # {"recall":[@5,@10,@20], "ndcg":[@5,@10,@20]} at best epoch
        "val_recall20_selected": at(val, "recall", 20),   # the actual selection score
        "val_ndcg10": at(val, "ndcg", 10),   # best-epoch (NOT running max); fixes prior bug
        "val_recall10": at(val, "recall", 10),
        "test_ndcg10": at(test, "ndcg", 10),
        "test_recall10": at(test, "recall", 10),
        "log": str(logs[-1].relative_to(REPO)),
        "wall_time_s": dt,
        "_note": "single explicit-hparam run (NOT part of the official 54-point "
                 "grid_selection.json unless --out points there). val_*/test_* are "
                 "taken AT the best (selected) epoch, the checkpoint RLMRec saves; "
                 "selection is RLMRec-native Recall@20, not NDCG@10.",
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[r1_single] done in {dt/60:.1f} min — best_epoch={best_epoch} | "
          f"val NDCG@10={at(val,'ndcg',10)} (sel Recall@20={at(val,'recall',20)}) | "
          f"test NDCG@10={at(test,'ndcg',10)}")


if __name__ == "__main__":
    main()
