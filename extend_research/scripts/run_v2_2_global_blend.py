from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_v2_1_refined_router import best_expert, candidate_weights, metric_tuple
from train_v2_router import evaluate_baselines, evaluate_blend_cache, load_cache, prepare_blend_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2.2 dataset-level global blend-router experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_2_router.json"))
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
    selected_rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights_grid = candidate_weights(dataset["grid"])
        print(f"=== {dataset['name']} ({len(weights_grid)} weights) ===", flush=True)
        seed_payloads = load_seed_payloads(
            cache_dir=cache_dir,
            dataset_slug=dataset["slug"],
            top_k=int(config["top_k"]),
            seeds=seeds,
            missing_score=float(config["model"]["missing_score"]),
        )
        if not seed_payloads:
            print(f"  no cache files found, skipping", flush=True)
            continue

        selected_weights, selected_val_metrics = select_global_weight(seed_payloads, weights_grid)
        selected_rows.append(
            {
                "dataset": dataset["name"],
                "dataset_slug": dataset["slug"],
                "n_seeds": len(seed_payloads),
                "weights": json.dumps(selected_weights),
                "mean_val_NDCG@10": selected_val_metrics["NDCG@10"],
                "mean_val_Recall@10": selected_val_metrics["Recall@10"],
                "mean_val_MRR": selected_val_metrics["MRR"],
            }
        )

        for payload in seed_payloads:
            test_metrics = evaluate_blend_cache(payload["test_rows"], selected_weights)
            test_baselines = payload["test_baselines"]
            test_best_name, test_best_metrics = best_expert(test_baselines)
            rows.append(
                {
                    "dataset": dataset["name"],
                    "dataset_slug": dataset["slug"],
                    "seed": payload["seed"],
                    "method": "dataset_global_blend",
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
            f"  selected weights={selected_weights} "
            f"mean_val_NDCG@10={selected_val_metrics['NDCG@10']:.6f}",
            flush=True,
        )

    summary_rows = summarize(rows)
    write_csv(output_dir / "v2_2_selected_weights.csv", selected_rows)
    write_csv(output_dir / "v2_2_router_by_seed.csv", rows)
    write_csv(output_dir / "v2_2_router_summary.csv", summary_rows)
    write_markdown(output_dir / "V2_2_Router_Results.md", selected_rows, summary_rows, rows)
    print(f"Wrote V2.2 router outputs to {output_dir}", flush=True)


def load_seed_payloads(
    cache_dir: Path,
    dataset_slug: str,
    top_k: int,
    seeds: list[int],
    missing_score: float,
) -> list[dict[str, Any]]:
    payloads = []
    for seed in seeds:
        val_path = cache_dir / dataset_slug / f"seed-{seed}" / f"val_top{top_k}.npz"
        test_path = cache_dir / dataset_slug / f"seed-{seed}" / f"test_top{top_k}.npz"
        if not val_path.exists() or not test_path.exists():
            print(f"  seed={seed}: missing cache, skipping", flush=True)
            continue
        val_cache = load_cache(val_path)
        test_cache = load_cache(test_path)
        payloads.append(
            {
                "seed": seed,
                "val_rows": prepare_blend_rows(val_cache, missing_score=missing_score),
                "test_rows": prepare_blend_rows(test_cache, missing_score=missing_score),
                "test_baselines": evaluate_baselines(test_cache),
            }
        )
    return payloads


def select_global_weight(
    seed_payloads: list[dict[str, Any]],
    weights_grid: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], dict[str, float]]:
    best_weights = weights_grid[0]
    best_metrics = {"NDCG@10": -1.0, "Recall@10": -1.0, "MRR": -1.0}
    for weights in weights_grid:
        metrics_by_seed = [evaluate_blend_cache(payload["val_rows"], weights) for payload in seed_payloads]
        mean_metrics = {
            key: float(np.mean([metrics[key] for metrics in metrics_by_seed]))
            for key in ["NDCG@10", "Recall@10", "MRR", "HR@10"]
        }
        if metric_tuple(mean_metrics) > metric_tuple(best_metrics):
            best_weights = weights
            best_metrics = mean_metrics
    return best_weights, best_metrics


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
                "method": "dataset_global_blend",
                "n_seeds": len(items),
                "weights": items[0]["weights"],
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


def write_markdown(
    path: Path,
    selected_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# V2.2 Dataset-Level Global Blend Results",
        "",
        "## Selected Weights",
        "",
        "| Dataset | Seeds | Weights | Mean Val NDCG@10 |",
        "|---|---:|---|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_seeds']} | `{row['weights']}` | "
            f"{row['mean_val_NDCG@10']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Dataset | Weights | NDCG@10 | Best Expert | Delta | Recall@10 | MRR |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | `{row['weights']}` | {row['mean_NDCG@10']:.6f} | "
            f"{row['mean_test_best_expert_NDCG@10']:.6f} | "
            f"{row['mean_delta_vs_test_best_expert']:.6f} | "
            f"{row['mean_Recall@10']:.6f} | {row['mean_MRR']:.6f} |"
        )

    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Weights | Test NDCG@10 | Test Best | Delta |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | `{row['weights']}` | "
            f"{row['test_NDCG@10']:.6f} | "
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
