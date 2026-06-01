from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_v2_router import (
    EXPERT_KEYS,
    evaluate_baselines,
    evaluate_blend_cache,
    load_cache,
    prepare_blend_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2.1 refined and guarded blend-router experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_1_router.json"))
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
    thresholds = [float(value) for value in config["fallback_min_val_gains"]]
    rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights_grid = candidate_weights(dataset["grid"])
        print(f"=== {dataset['name']} ({len(weights_grid)} weights) ===", flush=True)
        for seed in seeds:
            val_path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"val_top{config['top_k']}.npz"
            test_path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"test_top{config['top_k']}.npz"
            if not val_path.exists() or not test_path.exists():
                print(f"  seed={seed}: missing cache, skipping", flush=True)
                continue

            val_cache = load_cache(val_path)
            test_cache = load_cache(test_path)
            val_baselines = evaluate_baselines(val_cache)
            test_baselines = evaluate_baselines(test_cache)
            val_best_name, val_best_metrics = best_expert(val_baselines)
            test_best_name, test_best_metrics = best_expert(test_baselines)

            val_rows = prepare_blend_rows(
                val_cache,
                missing_score=float(config["model"]["missing_score"]),
            )
            test_rows = prepare_blend_rows(
                test_cache,
                missing_score=float(config["model"]["missing_score"]),
            )
            best_weights, refined_val_metrics = select_weights(val_rows, weights_grid)
            refined_test_metrics = evaluate_blend_cache(test_rows, best_weights)
            val_gain = refined_val_metrics["NDCG@10"] - val_best_metrics["NDCG@10"]

            rows.append(
                result_row(
                    dataset=dataset,
                    seed=seed,
                    method="refined_blend",
                    selected_policy="blend",
                    selected_expert="",
                    weights=best_weights,
                    val_metrics=refined_val_metrics,
                    test_metrics=refined_test_metrics,
                    val_best_name=val_best_name,
                    val_best_metrics=val_best_metrics,
                    test_best_name=test_best_name,
                    test_best_metrics=test_best_metrics,
                    test_baselines=test_baselines,
                )
            )

            for threshold in thresholds:
                use_blend = val_gain >= threshold
                if use_blend:
                    final_metrics = refined_test_metrics
                    selected_policy = "blend"
                    selected_expert = ""
                    selected_weights = best_weights
                    final_val_metrics = refined_val_metrics
                else:
                    final_metrics = test_baselines[val_best_name]
                    selected_policy = "fallback"
                    selected_expert = val_best_name
                    selected_weights = expert_weights(val_best_name)
                    final_val_metrics = val_best_metrics

                rows.append(
                    result_row(
                        dataset=dataset,
                        seed=seed,
                        method=f"guarded_gain_{threshold:g}",
                        selected_policy=selected_policy,
                        selected_expert=selected_expert,
                        weights=selected_weights,
                        val_metrics=final_val_metrics,
                        test_metrics=final_metrics,
                        val_best_name=val_best_name,
                        val_best_metrics=val_best_metrics,
                        test_best_name=test_best_name,
                        test_best_metrics=test_best_metrics,
                        test_baselines=test_baselines,
                    )
                )

            print(
                f"  seed={seed}: val_gain={val_gain:.6f} weights={best_weights} "
                f"refined_test={refined_test_metrics['NDCG@10']:.6f} "
                f"test_best={test_best_name} {test_best_metrics['NDCG@10']:.6f}",
                flush=True,
            )

    summary_rows = summarize(rows)
    write_csv(output_dir / "v2_1_router_by_seed.csv", rows)
    write_csv(output_dir / "v2_1_router_summary.csv", summary_rows)
    write_markdown(output_dir / "V2_1_Router_Results.md", summary_rows, rows)
    print(f"Wrote V2.1 router outputs to {output_dir}", flush=True)


def candidate_weights(grid: dict[str, Any]) -> list[tuple[float, float, float]]:
    step = float(grid["step"])
    mins = [float(value) for value in grid["min"]]
    maxs = [float(value) for value in grid["max"]]
    scale = int(round(1.0 / step))
    weights: set[tuple[float, float, float]] = set()

    for a in range(scale + 1):
        for b in range(scale + 1 - a):
            c = scale - a - b
            candidate = (a / scale, b / scale, c / scale)
            if all(mins[i] - 1e-9 <= candidate[i] <= maxs[i] + 1e-9 for i in range(3)):
                weights.add(round_weights(candidate))

    for expert in EXPERT_KEYS:
        weights.add(expert_weights(expert))
    return sorted(weights)


def select_weights(
    prepared_rows: list[tuple[np.ndarray, np.ndarray, int, float]],
    weights_grid: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], dict[str, float]]:
    best_weights = weights_grid[0]
    best_metrics = {"NDCG@10": -1.0, "Recall@10": -1.0, "MRR": -1.0}
    for weights in weights_grid:
        metrics = evaluate_blend_cache(prepared_rows, weights)
        if metric_tuple(metrics) > metric_tuple(best_metrics):
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["NDCG@10"], metrics["Recall@10"], metrics["MRR"])


def best_expert(metrics_by_expert: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    return max(metrics_by_expert.items(), key=lambda item: metric_tuple(item[1]))


def expert_weights(expert: str) -> tuple[float, float, float]:
    return tuple(1.0 if key == expert else 0.0 for key in EXPERT_KEYS)  # type: ignore[return-value]


def round_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(value, 10) for value in weights)  # type: ignore[return-value]


def result_row(
    dataset: dict[str, Any],
    seed: int,
    method: str,
    selected_policy: str,
    selected_expert: str,
    weights: tuple[float, float, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    val_best_name: str,
    val_best_metrics: dict[str, float],
    test_best_name: str,
    test_best_metrics: dict[str, float],
    test_baselines: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "method": method,
        "selected_policy": selected_policy,
        "selected_expert": selected_expert,
        "weights": json.dumps(weights),
        "val_NDCG@10": val_metrics["NDCG@10"],
        "val_best_expert": val_best_name,
        "val_best_expert_NDCG@10": val_best_metrics["NDCG@10"],
        "val_gain_vs_best_expert": val_metrics["NDCG@10"] - val_best_metrics["NDCG@10"],
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["dataset_slug"], row["method"]), []).append(row)

    out = []
    for (dataset, dataset_slug, method), items in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "method": method,
                "n_seeds": len(items),
                "blend_or_router_selections": sum(1 for item in items if item["selected_policy"] == "blend"),
                "fallback_selections": sum(1 for item in items if item["selected_policy"] == "fallback"),
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
    return out


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
        "# V2.1 Refined Guarded Router Results",
        "",
        "| Dataset | Method | Seeds | Blend | Fallback | NDCG@10 | Best Expert | Delta | Recall@10 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n_seeds']} | "
            f"{row['blend_or_router_selections']} | {row['fallback_selections']} | "
            f"{row['mean_NDCG@10']:.6f} | {row['mean_test_best_expert_NDCG@10']:.6f} | "
            f"{row['mean_delta_vs_test_best_expert']:.6f} | "
            f"{row['mean_Recall@10']:.6f} | {row['mean_MRR']:.6f} |"
        )

    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Method | Policy | Weights | Val Gain | Test NDCG@10 | Test Best | Delta |",
            "|---|---:|---|---|---|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['method']} | "
            f"{row['selected_policy']} | `{row['weights']}` | "
            f"{row['val_gain_vs_best_expert']:.6f} | {row['test_NDCG@10']:.6f} | "
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
