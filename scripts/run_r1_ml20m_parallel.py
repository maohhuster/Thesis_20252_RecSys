#!/usr/bin/env python3
"""Run N R1 = RLMRec-gene ML-20M seeds in parallel on a single GPU.

Designed for Colab Pro L4 (and similar single-GPU setups): a single R1 d=128
ML-20M run on an L4 uses ~20% GPU compute and ~5 GB VRAM, leaving the device
4-5× under-utilised. Running multiple seeds concurrently saturates the device
and finishes all of them in roughly one wall-clock-equivalent of a single
seed — same total Compute Units, much shorter calendar time.

Mechanism
---------
The shared `lightgcn_gene.yml` would race if N processes patched it
simultaneously (each seed needs its own `ckpt_out_dir`). To avoid that, this
script builds a per-seed *isolated cwd* — a sibling RLMRec working directory
in which only `encoder/config/` is a real copy (so the yml can be patched
independently) and everything else (encoder/trainer code, data pickles,
models, etc.) is a symlink back to the canonical RLMRec source. Result: each
process has its own yml + its own logs + its own native RLMRec checkpoint
output dir, but the 173 MB data pickles are NOT duplicated.

Each subprocess writes its own stdout log under
`<ckpt_root>/seed-<N>/stdout.log` so you can `tail -f` each individually.

Usage on Colab Pro L4 (5 seeds in parallel):
    !python3 scripts/run_r1_ml20m_parallel.py \
        --ckpt_root /content/drive/MyDrive/r1_ckpt \
        --seeds 42 123 456 789 2026 \
        --embedding_size 128 --batch_size 2048

Lower `--batch_size` to 2048 (or 1024) if 5×4096-batch processes OOM the L4.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
RLMREC = REPO / "code" / "benchmark" / "external" / "RLMRec"

WINNER = dict(
    layer_num=3,
    mask_ratio=0.10,
    recon_weight=0.10,
    re_temperature=0.2,
    reg_weight=1e-7,
)


def setup_seed_cwd(seed: int, hparams: dict, ckpt_dir: Path,
                   work_root: Path) -> Path:
    """Build a per-seed isolated RLMRec cwd.

    Only `encoder/config/` is a real copy (so the per-seed yml patch is
    independent); everything else under RLMREC is symlinked, so the 173 MB
    of data pickles is NOT duplicated.
    """
    work = work_root / f"seed-{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    for item in RLMREC.iterdir():
        target = work / item.name
        if item.name == "encoder":
            target.mkdir()
            for sub in item.iterdir():
                if sub.name == "config":
                    # Real copy — we patch the yml inside.
                    shutil.copytree(sub, target / "config")
                else:
                    (target / sub.name).symlink_to(sub.resolve())
        else:
            target.symlink_to(item.resolve())

    yml = work / "encoder" / "config" / "modelconf" / "lightgcn_gene.yml"
    cfg = yaml.safe_load(yml.read_text())
    cfg["model"]["embedding_size"] = int(hparams["embedding_size"])
    cfg["model"]["ml20m"] = {
        "layer_num": int(hparams["layer_num"]),
        "reg_weight": float(hparams["reg_weight"]),
        "mask_ratio": float(hparams["mask_ratio"]),
        "recon_weight": float(hparams["recon_weight"]),
        "re_temperature": float(hparams["re_temperature"]),
    }
    cfg["train"]["batch_size"] = int(hparams["batch_size"])
    cfg["train"]["test_step"] = int(hparams["eval_every"])
    cfg["train"]["ckpt_out_dir"] = str(ckpt_dir)
    cfg["train"]["resume"] = bool(hparams["resume"])
    yml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return work


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 2026],
                   help="seeds to run in parallel (default: all 5 paper seeds)")
    p.add_argument("--ckpt_root", required=True,
                   help="<ckpt_root>/seed-<N>/ becomes each seed's checkpoint "
                        "+ stdout-log dir. On Colab, point this at Drive.")
    p.add_argument("--embedding_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4096,
                   help="lower to 2048/1024 if 5 parallel runs OOM the GPU")
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--layer_num", type=int, default=WINNER["layer_num"])
    p.add_argument("--mask_ratio", type=float, default=WINNER["mask_ratio"])
    p.add_argument("--recon_weight", type=float, default=WINNER["recon_weight"])
    p.add_argument("--re_temperature", type=float,
                   default=WINNER["re_temperature"])
    p.add_argument("--reg_weight", type=float, default=WINNER["reg_weight"])
    p.add_argument("--resume", action="store_true",
                   help="each subprocess resumes from its own seed-N/training_state.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda", default="0", help="CUDA index (all seeds share the same GPU)")
    p.add_argument("--stagger_seconds", type=int, default=8,
                   help="seconds between launches; avoids data-loading thrash")
    p.add_argument("--work_root", default=None,
                   help="parent dir for per-seed cwds (default: tempfile.mkdtemp)")
    p.add_argument("--keep_work", action="store_true",
                   help="don't delete per-seed cwds after run (debug)")
    args = p.parse_args()

    hparams = dict(
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        layer_num=args.layer_num,
        mask_ratio=args.mask_ratio,
        recon_weight=args.recon_weight,
        re_temperature=args.re_temperature,
        reg_weight=args.reg_weight,
        resume=args.resume,
    )
    work_root = (Path(args.work_root) if args.work_root
                 else Path(tempfile.mkdtemp(prefix="rlmrec-parallel-")))
    work_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root).resolve()

    print(f"[parallel] launching {len(args.seeds)} seeds: {args.seeds}")
    print(f"[parallel] work_root  = {work_root}")
    print(f"[parallel] ckpt_root  = {ckpt_root}")
    print(f"[parallel] hparams    = d={args.embedding_size} bs={args.batch_size} "
          f"ln={args.layer_num} mr={args.mask_ratio} rw={args.recon_weight} "
          f"rt={args.re_temperature} reg={args.reg_weight} "
          f"eval_every={args.eval_every} resume={args.resume}")

    procs = []
    for seed in args.seeds:
        ckpt_dir = ckpt_root / f"seed-{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        cwd = setup_seed_cwd(seed, hparams, ckpt_dir, work_root)
        log_path = ckpt_dir / "stdout.log"
        cmd = [
            sys.executable, "encoder/train_encoder.py",
            "--model", "lightgcn_gene", "--dataset", "ml20m",
            "--device", args.device, "--cuda", args.cuda,
            "--seed", str(seed),
        ]
        print(f"[parallel] seed={seed:5d}  cwd={cwd}  log={log_path}")
        log_f = log_path.open("w")
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs.append((seed, proc, log_f, log_path, cwd))
        time.sleep(args.stagger_seconds)

    print(f"[parallel] all {len(procs)} subprocesses launched; waiting…")
    print(f"[parallel] watch progress in another cell:  !tail -f {ckpt_root}/seed-*/stdout.log")

    t0 = time.time()
    failed = []
    for seed, proc, log_f, log_path, cwd in procs:
        rc = proc.wait()
        log_f.close()
        wall = (time.time() - t0) / 60.0
        if rc == 0:
            print(f"[parallel] seed={seed:5d}  DONE   exit=0   (wall {wall:.1f} min from kickoff)")
        else:
            print(f"[parallel] seed={seed:5d}  FAILED exit={rc}  log={log_path}")
            failed.append(seed)

    if not args.keep_work:
        for _, _, _, _, cwd in procs:
            shutil.rmtree(cwd, ignore_errors=True)
        try:
            work_root.rmdir()
        except OSError:
            pass  # not empty / not tmp

    if failed:
        print(f"[parallel] {len(failed)}/{len(args.seeds)} seeds failed: {failed}")
        sys.exit(1)
    print(f"[parallel] ✓ all {len(args.seeds)} seeds completed successfully")
    print(f"[parallel]   per-seed checkpoints + logs: {ckpt_root}/seed-{{N}}/")


if __name__ == "__main__":
    main()
