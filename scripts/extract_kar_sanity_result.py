#!/usr/bin/env python3
"""Parse KAR sanity-check log and emit code/benchmark/external/KAR_upstream/sanity_result.json.

Usage:
    python3 scripts/extract_kar_sanity_result.py /tmp/kar_ctr.log
"""
import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULT_PATH = REPO / "code" / "benchmark" / "external" / "KAR_upstream" / "sanity_result.json"
UPSTREAM = REPO / "code" / "benchmark" / "external" / "KAR_upstream" / "upstream"

def main(log_path: str):
    text = Path(log_path).read_text()
    text = text.replace("\r", "\n")

    # Per-epoch lines look like:
    # EPOCH 0  STEP 734 train loss: 0.55243, train time: 32.12, test loss: 0.50213,
    #          test time: 1.83, auc: 0.78421, logloss: 0.50211
    epoch_pat = re.compile(
        r"EPOCH\s+(\d+).*?train loss:\s*([\d.]+).*?train time:\s*([\d.]+).*?"
        r"test loss:\s*([\d.]+).*?test time:\s*([\d.]+).*?"
        r"auc:\s*([\d.]+).*?logloss:\s*([\d.]+)",
        re.DOTALL,
    )
    rows = [
        {
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "train_time_s": float(m.group(3)),
            "test_loss": float(m.group(4)),
            "test_time_s": float(m.group(5)),
            "test_auc": float(m.group(6)),
            "test_logloss": float(m.group(7)),
        }
        for m in epoch_pat.finditer(text)
    ]
    if not rows:
        print(f"ERROR: no EPOCH lines found in {log_path}", file=sys.stderr)
        sys.exit(1)

    best = max(rows, key=lambda r: r["test_auc"])

    # Upstream commit
    sha = ""
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(UPSTREAM), text=True
        ).strip()
    except Exception:
        pass

    published = {
        "DIN_KAR_AUC": 0.8143, "DIN_KAR_LogLoss": 0.5096,
        "DIN_baseline_AUC": 0.7878, "DIN_baseline_LogLoss": 0.5364,
    }

    delta_auc = best["test_auc"] - published["DIN_KAR_AUC"]
    delta_ll = best["test_logloss"] - published["DIN_KAR_LogLoss"]

    out = {
        "upstream_repo": "YunjiaXi/Open-World-Knowledge-Augmented-Recommendation",
        "upstream_commit": sha,
        "config": "DIN + KAR (HEA, augment=True), bs=1024, lr=5e-4, "
                  "export_num=2, specific_export_num=5, epochs=5, device=mps",
        "dataset": "ml-1m (native CTR)",
        "per_epoch": rows,
        "best_epoch": best["epoch"],
        "our_run_test_AUC": best["test_auc"],
        "our_run_test_LogLoss": best["test_logloss"],
        "our_run_train_time_s_per_epoch_mean": sum(r["train_time_s"] for r in rows) / len(rows),
        "published_DIN_KAR": published,
        "auc_delta_vs_published": delta_auc,
        "logloss_delta_vs_published": delta_ll,
        "sanity_check_passed": abs(delta_auc) < 0.020,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kar_ctr.log"
    main(log)
