from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analyze_v2_2_segments import METRIC_KEYS, ground_truth_items, rank_blend, read_selected_weights, user_metrics
from train_v2_router import EXPERT_KEYS, load_cache


METHODS = ["global_blend", *EXPERT_KEYS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2.4 per-user oracle gap analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_4_oracle_gap.json"))
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
    selected_weights = read_selected_weights(resolve_path(config["selected_weights_path"], config_path))

    selected_datasets = set(args.datasets) if args.datasets else None
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    by_seed_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights = selected_weights[dataset["slug"]]
        print(f"=== {dataset['name']} weights={weights} ===", flush=True)
        for seed in seeds:
            path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"test_top{config['top_k']}.npz"
            if not path.exists():
                print(f"  seed={seed}: missing cache, skipping", flush=True)
                continue

            cache = load_cache(path)
            result = evaluate_oracles(
                cache=cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
            )
            by_seed_rows.append(seed_row(dataset, seed, weights, result))
            selection_rows.extend(selection_count_rows(dataset, seed, result))
            print(
                f"  seed={seed}: blend={result['global_blend']['NDCG@10']:.6f} "
                f"oracle_all={result['oracle_all']['NDCG@10']:.6f}",
                flush=True,
            )

    summary_rows = summarize(by_seed_rows)
    write_csv(output_dir / "oracle_gap_by_seed.csv", by_seed_rows)
    write_csv(output_dir / "oracle_selection_counts.csv", selection_rows)
    write_csv(output_dir / "oracle_gap_summary.csv", summary_rows)
    write_markdown(output_dir / "V2_4_Oracle_Gap.md", summary_rows, by_seed_rows)
    print(f"Wrote V2.4 oracle outputs to {output_dir}", flush=True)


def evaluate_oracles(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
) -> dict[str, Any]:
    per_method: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in METRIC_KEYS}
        for method in METHODS
    }
    oracle_expert = {metric: [] for metric in METRIC_KEYS}
    oracle_all = {metric: [] for metric in METRIC_KEYS}
    oracle_expert_selection: list[str] = []
    oracle_all_selection: list[str] = []

    for user_idx in range(len(cache["users"])):
        gt_items = ground_truth_items(cache, user_idx)
        user_method_metrics = {
            "global_blend": user_metrics(rank_blend(cache, user_idx, weights, missing_score), gt_items),
            "m7": user_metrics(cache["m7_top_items"][user_idx], gt_items),
            "r1": user_metrics(cache["r1_top_items"][user_idx], gt_items),
            "r1plus": user_metrics(cache["r1plus_top_items"][user_idx], gt_items),
        }
        for method, metrics in user_method_metrics.items():
            for metric in METRIC_KEYS:
                per_method[method][metric].append(metrics[metric])

        expert_choice = choose_best(["m7", "r1", "r1plus"], user_method_metrics)
        all_choice = choose_best(METHODS, user_method_metrics)
        oracle_expert_selection.append(expert_choice)
        oracle_all_selection.append(all_choice)
        for metric in METRIC_KEYS:
            oracle_expert[metric].append(user_method_metrics[expert_choice][metric])
            oracle_all[metric].append(user_method_metrics[all_choice][metric])

    mean_metrics = {
        method: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for method, metrics in per_method.items()
    }
    mean_metrics["oracle_expert"] = {
        metric: float(np.mean(values)) for metric, values in oracle_expert.items()
    }
    mean_metrics["oracle_all"] = {
        metric: float(np.mean(values)) for metric, values in oracle_all.items()
    }
    best_fixed = choose_best(["m7", "r1", "r1plus"], mean_metrics)
    mean_metrics["best_fixed"] = mean_metrics[best_fixed]
    return {
        **mean_metrics,
        "best_fixed_name": best_fixed,
        "oracle_expert_selection": Counter(oracle_expert_selection),
        "oracle_all_selection": Counter(oracle_all_selection),
        "n_users": len(cache["users"]),
    }


def choose_best(methods: list[str], metrics_by_method: dict[str, dict[str, float]]) -> str:
    return max(
        methods,
        key=lambda method: (
            metrics_by_method[method]["NDCG@10"],
            metrics_by_method[method]["Recall@10"],
            metrics_by_method[method]["MRR"],
        ),
    )


def seed_row(
    dataset: dict[str, Any],
    seed: int,
    weights: tuple[float, float, float],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "n_users": result["n_users"],
        "weights": json.dumps(weights),
        "best_fixed": result["best_fixed_name"],
        "best_fixed_NDCG@10": result["best_fixed"]["NDCG@10"],
        "global_blend_NDCG@10": result["global_blend"]["NDCG@10"],
        "oracle_expert_NDCG@10": result["oracle_expert"]["NDCG@10"],
        "oracle_all_NDCG@10": result["oracle_all"]["NDCG@10"],
        "blend_delta_vs_best_fixed": result["global_blend"]["NDCG@10"] - result["best_fixed"]["NDCG@10"],
        "oracle_expert_gap_vs_best_fixed": result["oracle_expert"]["NDCG@10"] - result["best_fixed"]["NDCG@10"],
        "oracle_all_gap_vs_blend": result["oracle_all"]["NDCG@10"] - result["global_blend"]["NDCG@10"],
        "oracle_all_gap_vs_best_fixed": result["oracle_all"]["NDCG@10"] - result["best_fixed"]["NDCG@10"],
        "best_fixed_Recall@10": result["best_fixed"]["Recall@10"],
        "global_blend_Recall@10": result["global_blend"]["Recall@10"],
        "oracle_all_Recall@10": result["oracle_all"]["Recall@10"],
        "best_fixed_MRR": result["best_fixed"]["MRR"],
        "global_blend_MRR": result["global_blend"]["MRR"],
        "oracle_all_MRR": result["oracle_all"]["MRR"],
    }


def selection_count_rows(dataset: dict[str, Any], seed: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for oracle_name, counter_key in [
        ("oracle_expert", "oracle_expert_selection"),
        ("oracle_all", "oracle_all_selection"),
    ]:
        total = sum(result[counter_key].values())
        for method, count in sorted(result[counter_key].items()):
            rows.append(
                {
                    "dataset": dataset["name"],
                    "dataset_slug": dataset["slug"],
                    "seed": seed,
                    "oracle": oracle_name,
                    "selected_method": method,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    return rows


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
                "n_seeds": len(items),
                "mean_best_fixed_NDCG@10": mean(items, "best_fixed_NDCG@10"),
                "mean_global_blend_NDCG@10": mean(items, "global_blend_NDCG@10"),
                "mean_oracle_expert_NDCG@10": mean(items, "oracle_expert_NDCG@10"),
                "mean_oracle_all_NDCG@10": mean(items, "oracle_all_NDCG@10"),
                "mean_blend_delta_vs_best_fixed": mean(items, "blend_delta_vs_best_fixed"),
                "mean_oracle_expert_gap_vs_best_fixed": mean(items, "oracle_expert_gap_vs_best_fixed"),
                "mean_oracle_all_gap_vs_blend": mean(items, "oracle_all_gap_vs_blend"),
                "mean_oracle_all_gap_vs_best_fixed": mean(items, "oracle_all_gap_vs_best_fixed"),
                "mean_global_blend_Recall@10": mean(items, "global_blend_Recall@10"),
                "mean_oracle_all_Recall@10": mean(items, "oracle_all_Recall@10"),
                "mean_global_blend_MRR": mean(items, "global_blend_MRR"),
                "mean_oracle_all_MRR": mean(items, "oracle_all_MRR"),
            }
        )
    return summary


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V2.4 Oracle Gap",
        "",
        "| Dataset | Best Fixed | V2.2 Blend | Oracle Expert | Oracle All | Blend Gain | Oracle-All Gap vs Blend |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_best_fixed_NDCG@10']:.6f} | "
            f"{row['mean_global_blend_NDCG@10']:.6f} | "
            f"{row['mean_oracle_expert_NDCG@10']:.6f} | "
            f"{row['mean_oracle_all_NDCG@10']:.6f} | "
            f"{row['mean_blend_delta_vs_best_fixed']:.6f} | "
            f"{row['mean_oracle_all_gap_vs_blend']:.6f} |"
        )

    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Best Fixed | V2.2 Blend | Oracle All | Blend Gain | Oracle-All Gap vs Blend |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['best_fixed']} | "
            f"{row['global_blend_NDCG@10']:.6f} | {row['oracle_all_NDCG@10']:.6f} | "
            f"{row['blend_delta_vs_best_fixed']:.6f} | "
            f"{row['oracle_all_gap_vs_blend']:.6f} |"
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
