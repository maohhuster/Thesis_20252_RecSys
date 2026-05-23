#!/usr/bin/env python3
"""Single, resumable R1 (RLMRec-gene) ML-20M run — NOT the sweep.

Built for the Colab d=128 deployment run. Exposes the knobs the upstream
RLMRec path hides and makes long runs crash-safe:

  * --embedding_size  (default 128: capacity-matched to the M7/LightGCN-SF
                        backbone; the paper's deployed R1 ML-20M dimension)
  * --batch_size      (default 4096: RLMRec upstream default; lower it if a
                        Colab T4/L4 OOMs at d=128)
  * resumable         best_model.pt + training_state.pt are written into the
                        project checkpoint dir EVERY evaluation; pass --resume
                        to continue an interrupted run from the last state
                        (model + optimizer + epoch + best + patience + RNG).
  * eval printing      RLMRec already prints `Epoch N Validation set [...]`
                        every test_step (default 3 epochs, --eval_every);
                        unchanged here.

Selection / early-stopping stays RLMRec-native (best validation Recall@20,
patience 5) — identical to the official grid, so this run is comparable to
the 54-pt selection. The committed lightgcn_gene.yml is ALWAYS restored
verbatim (try/finally); the trainer's checkpoint/resume code is opt-in and
inert for every other RLMRec path (gated on train.ckpt_out_dir).

Examples
--------
First run (d=128, committed grid winner, seed 42):
    python3 scripts/run_r1_ml20m_resumable.py --device cuda --seed 42

Resume after a Colab disconnect (same command + --resume):
    python3 scripts/run_r1_ml20m_resumable.py --device cuda --seed 42 --resume

Lower batch size if the GPU OOMs at d=128:
    python3 scripts/run_r1_ml20m_resumable.py --device cuda --seed 42 \
        --batch_size 2048 --resume
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r1_ml20m_single as R1  # verified parse_log / pick_device / paths

REPO, RLMREC, YML, LOG_DIR, DATASET = (
    R1.REPO, R1.RLMREC, R1.YML, R1.LOG_DIR, R1.DATASET)

# committed 54-pt grid winner (selected at d=32; re-confirmed at d=128 by the
# reduced re-validation grid) — the cell deployed for the paper.
WINNER = dict(layer_num=3, mask_ratio=0.10, recon_weight=0.10,
              re_temperature=0.2, reg_weight=1e-7)


def patch_yml_full(emb, bs, cell, ckpt_dir, resume, eval_every):
    """Dict round-trip patch (configurator uses yaml.safe_load, so safe_dump
    is faithful). Caller restores the original text verbatim in finally."""
    cfg = yaml.safe_load(YML.read_text())
    cfg["model"]["embedding_size"] = int(emb)
    cfg["model"]["ml20m"] = {
        "layer_num": int(cell["layer_num"]),
        "reg_weight": float(cell["reg_weight"]),
        "mask_ratio": float(cell["mask_ratio"]),
        "recon_weight": float(cell["recon_weight"]),
        "re_temperature": float(cell["re_temperature"]),
    }
    cfg["train"]["batch_size"] = int(bs)
    cfg["train"]["test_step"] = int(eval_every)
    cfg["train"]["ckpt_out_dir"] = str(ckpt_dir)   # activates resumable path
    cfg["train"]["resume"] = bool(resume)
    YML.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", help="cuda|cpu|mps|auto")
    p.add_argument("--resume", action="store_true",
                   help="continue from the last training_state.pt in --checkpoint_dir")
    p.add_argument("--eval_every", type=int, default=3,
                   help="RLMRec test_step: evaluate+print+checkpoint every N epochs")
    p.add_argument("--layer_num", type=int, default=WINNER["layer_num"])
    p.add_argument("--mask_ratio", type=float, default=WINNER["mask_ratio"])
    p.add_argument("--recon_weight", type=float, default=WINNER["recon_weight"])
    p.add_argument("--re_temperature", type=float, default=WINNER["re_temperature"])
    p.add_argument("--reg_weight", type=float, default=WINNER["reg_weight"])
    p.add_argument("--checkpoint_dir", default=None,
                   help="default: code/benchmark/checkpoints/r1/bge-large-en-v1.5/seed-<seed>")
    p.add_argument("--out", default=None, help="result JSON (default under code/benchmark/hparams/r1/)")
    a = p.parse_args()

    device = R1.pick_device(a.device)
    ckpt_dir = (Path(a.checkpoint_dir) if a.checkpoint_dir else
                REPO / "code" / "benchmark" / "checkpoints" / "r1"
                / "bge-large-en-v1.5" / f"seed-{a.seed}")
    ckpt_dir = ckpt_dir.resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(a.out) if a.out else
                REPO / "code" / "benchmark" / "hparams" / "r1"
                / f"resumable_d{a.embedding_size}_seed{a.seed}.json")
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell = dict(layer_num=a.layer_num, mask_ratio=a.mask_ratio,
                recon_weight=a.recon_weight, re_temperature=a.re_temperature,
                reg_weight=a.reg_weight)
    print(f"[r1_resumable] ml20m d={a.embedding_size} bs={a.batch_size} "
          f"seed={a.seed} device={device} resume={a.resume}")
    print(f"[r1_resumable] cell={cell}")
    print(f"[r1_resumable] ckpt_dir={ckpt_dir}")
    print(f"[r1_resumable]   -> best_model.pt + training_state.pt written every "
          f"{a.eval_every} epochs (selection: RLMRec-native val Recall@20)")

    original = YML.read_text()
    t0 = time.time()
    try:
        patch_yml_full(a.embedding_size, a.batch_size, cell, ckpt_dir,
                        a.resume, a.eval_every)
        cmd = [sys.executable, "encoder/train_encoder.py",
               "--model", "lightgcn_gene", "--dataset", DATASET,
               "--device", device, "--seed", str(a.seed)]
        print(f"[r1_resumable] -> {' '.join(cmd)}  (cwd={RLMREC})")
        import subprocess
        r = subprocess.run(cmd, cwd=str(RLMREC))
    finally:
        YML.write_text(original)  # never leave the committed yml mutated
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[r1_resumable] FAILED (rc={r.returncode}) after {dt/60:.1f} min — "
              f"re-run with --resume to continue from the last checkpoint")
        sys.exit(1)

    logs = sorted(LOG_DIR.glob("*.log"), key=lambda q: q.stat().st_mtime)
    logs = [q for q in logs if q.stat().st_mtime >= t0 - 1]
    best_epoch = val = test = None
    if logs:
        best_epoch, val, test = R1.parse_log(logs[-1])

    def at(d, metric, k):
        if not d:
            return None
        v = d.get(metric)
        idx = {5: 0, 10: 1, 20: 2}[k]
        return v[idx] if v and len(v) > idx else None

    out = {
        "method": "R1 = RLMRec-gene (resumable single run)",
        "dataset": DATASET, "seed": a.seed, "device": device,
        "embedding_size": a.embedding_size, "batch_size": a.batch_size,
        **cell,
        "best_epoch": best_epoch,
        "selection_metric": "Recall@20 (RLMRec upstream-native)",
        "val": val, "test": test,
        "val_recall20_selected": at(val, "recall", 20),
        "val_ndcg10": at(val, "ndcg", 10),
        "test_ndcg10": at(test, "ndcg", 10),
        "test_recall10": at(test, "recall", 10),
        "checkpoint_dir": str(ckpt_dir.relative_to(REPO)
                              if ckpt_dir.is_relative_to(REPO) else ckpt_dir),
        "best_model_pt": (ckpt_dir / "best_model.pt").exists(),
        "training_state_pt": (ckpt_dir / "training_state.pt").exists(),
        "log": str(logs[-1].relative_to(REPO)) if logs else None,
        "wall_time_s": dt,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[r1_resumable] done in {dt/60:.1f} min — best_epoch={best_epoch} | "
          f"test NDCG@10={at(test,'ndcg',10)} | "
          f"ckpt: best_model.pt={out['best_model_pt']} "
          f"training_state.pt={out['training_state_pt']}")


if __name__ == "__main__":
    main()
