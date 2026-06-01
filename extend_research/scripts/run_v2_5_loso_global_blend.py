from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_v2_1_refined_router import best_expert, candidate_weights
from run_v2_2_global_blend import load_seed_payloads, select_global_weight
from train_v2_router import evaluate_blend_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2.5 leave-one-seed-out global blend robustness check.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_5_loso_global_blend.json"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    config = read_json(config_path)
    cache_dir = resolve_path(config["candidate_cache_dir"], config_path)
    output_dir = resolve_path(config["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = set(args.datasets) if args.datasets else None
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights_grid = candidate_weights(dataset["grid"])
        print(f"=== {dataset['name']} ({len(weights_grid)} weights) ===", flush=True)
        all_payloads = load_seed_payloads(
            cache_dir=cache_dir,
            dataset_slug=dataset["slug"],
            top_k=int(config["top_k"]),
            seeds=seeds,
            missing_score=float(config["model"]["missing_score"]),
        )
        payload_by_seed = {int(payload["seed"]): payload for payload in all_payloads}

        for heldout_seed in seeds:
            if heldout_seed not in payload_by_seed:
                print(f"  seed={heldout_seed}: missing cache, skipping", flush=True)
                continue
            train_payloads = [
                payload for seed, payload in payload_by_seed.items()
                if seed != heldout_seed
            ]
            if not train_payloads:
                print(f"  seed={heldout_seed}: no training seeds, skipping", flush=True)
                continue

            selected_weights, selected_val_metrics = select_global_weight(train_payloads, weights_grid)
            test_payload = payload_by_seed[heldout_seed]
            test_metrics = evaluate_blend_cache(test_payload["test_rows"], selected_weights)
            test_baselines = test_payload["test_baselines"]
            test_best_name, test_best_metrics = best_expert(test_baselines)
            rows.append(
                {
                    "dataset": dataset["name"],
                    "dataset_slug": dataset["slug"],
                    "heldout_seed": heldout_seed,
                    "train_seeds": json.dumps([int(payload["seed"]) for payload in train_payloads]),
                    "weights": json.dumps(selected_weights),
                    "selection_mean_val_NDCG@10": selected_val_metrics["NDCG@10"],
                    "test_NDCG@10": test_metrics["NDCG@10"],
                    "test_Recall@10": test_metrics["Recall@10"],
                    "test_MRR": test_metrics["MRR"],
                    "test_best_expert": test_best_name,
                    "test_best_expert_NDCG@10": test_best_metrics["NDCG@10"],
                    "delta_vs_test_best_expert": test_metrics["NDCG@10"] - test_best_metrics["NDCG@10"],
                    "m7_NDCG@10": test_baselines["m7"]["NDCG@10"],
                    "r1_NDCG@10": test_baselines["r1"]["NDCG@10"],
                    "r1plus_NDCG@10": test_baselines["r1plus"]["NDCG@10"],
                }
            )
            print(
                f"  heldout={heldout_seed}: weights={selected_weights} "
                f"test={test_metrics['NDCG@10']:.6f} "
                f"best={test_best_name} {test_best_metrics['NDCG@10']:.6f}",
                flush=True,
            )

    summary_rows = summarize(rows)
    write_csv(output_dir / "v2_5_loso_by_seed.csv", rows)
    write_csv(output_dir / "v2_5_loso_summary.csv", summary_rows)
    write_markdown(output_dir / "V2_5_LOSO_Global_Blend.md", summary_rows, rows)
    print(f"Wrote V2.5 LOSO outputs to {output_dir}", flush=True)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["dataset_slug"]), []).append(row)

    summary = []
    for (dataset, dataset_slug), items in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "method": "loso_global_blend",
                "n_seeds": len(items),
                "unique_weights": ";".join(sorted(set(item["weights"] for item in items))),
                "mean_NDCG@10": float(np.mean([item["test_NDCG@10"] for item in items])),
                "std_NDCG@10": float(np.std([item["test_NDCG@10"] for item in items])),
                "mean_test_best_expert_NDCG@10": float(
                    np.mean([item["test_best_expert_NDCG@10"] for item in items])
                ),
                "mean_delta_vs_test_best_expert": float(
                    np.mean([item["delta_vs_test_best_expert"] for item in items])
                ),
                "mean_Recall@10": float(np.mean([item["test_Recall@10"] for item in items])),
                "mean_MRR": float(np.mean([item["test_MRR"] for item in items])),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V2.5 Leave-One-Seed-Out Global Blend",
        "",
        "| Dataset | Seeds | NDCG@10 | Best Expert | Delta | Unique Weights |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_seeds']} | {row['mean_NDCG@10']:.6f} | "
            f"{row['mean_test_best_expert_NDCG@10']:.6f} | "
            f"{row['mean_delta_vs_test_best_expert']:.6f} | "
            f"`{row['unique_weights']}` |"
        )

    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Held-Out Seed | Train Seeds | Weights | Test NDCG@10 | Test Best | Delta |",
            "|---|---:|---|---|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['heldout_seed']} | `{row['train_seeds']}` | "
            f"`{row['weights']}` | {row['test_NDCG@10']:.6f} | "
            f"{row['test_best_expert']} {row['test_best_expert_NDCG@10']:.6f} | "
            f"{row['delta_vs_test_best_expert']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_config(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def resolve_path(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent.parent / candidate).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
