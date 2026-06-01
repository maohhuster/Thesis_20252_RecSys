from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_v2_2_segments import METRIC_KEYS, ground_truth_items, read_selected_weights
from run_v3_action_router import (
    ACTIONS,
    EXPERT_KEYS,
    coefficient_rows,
    metrics_by_action,
    user_features,
)
from train_v2_router import load_cache


SWITCH_ACTIONS = ["m7", "r1", "r1plus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3.1 pairwise gain router.")
    parser.add_argument("--config", type=Path, default=Path("configs/v3_1_pairwise_gain_router.json"))
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
    action_rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights = selected_weights[dataset["slug"]]
        print(f"=== {dataset['name']} weights={weights} ===", flush=True)
        for seed in seeds:
            val_path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"val_top{config['top_k']}.npz"
            test_path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"test_top{config['top_k']}.npz"
            if not val_path.exists() or not test_path.exists():
                print(f"  seed={seed}: missing cache, skipping", flush=True)
                continue

            val_cache = load_cache(val_path)
            test_cache = load_cache(test_path)
            models, train_stats = train_gain_models(
                cache=val_cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
                alpha=float(config["model"]["ridge_alpha"]),
            )
            threshold, val_ndcg = select_threshold(
                cache=val_cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
                models=models,
                thresholds=[float(value) for value in config["model"]["gain_thresholds"]],
            )
            result = evaluate_gain_router(
                cache=test_cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
                models=models,
                threshold=threshold,
            )
            by_seed_rows.append(seed_row(dataset, seed, weights, result, train_stats, threshold, val_ndcg))
            action_rows.extend(action_count_rows(dataset, seed, result))
            print(
                f"  seed={seed}: threshold={threshold:.4f} "
                f"router={result['gain_router']['NDCG@10']:.6f} "
                f"blend={result['global_blend']['NDCG@10']:.6f} "
                f"delta={result['gain_router']['NDCG@10'] - result['global_blend']['NDCG@10']:.6f}",
                flush=True,
            )

    summary_rows = summarize(by_seed_rows)
    write_csv(output_dir / "v3_1_pairwise_gain_by_seed.csv", by_seed_rows)
    write_csv(output_dir / "v3_1_pairwise_gain_summary.csv", summary_rows)
    write_csv(output_dir / "v3_1_pairwise_gain_action_counts.csv", action_rows)
    write_markdown(output_dir / "V3_1_Pairwise_Gain_Router.md", summary_rows, by_seed_rows)
    print(f"Wrote V3.1 pairwise-gain outputs to {output_dir}", flush=True)


def train_gain_models(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = []
    targets = {action: [] for action in SWITCH_ACTIONS}
    for user_idx in range(len(cache["users"])):
        gt_items = ground_truth_items(cache, user_idx)
        action_metrics = metrics_by_action(cache, user_idx, weights, missing_score, gt_items)
        features.append(user_features(cache, user_idx))
        blend_ndcg = action_metrics["global_blend"]["NDCG@10"]
        for action in SWITCH_ACTIONS:
            targets[action].append(action_metrics[action]["NDCG@10"] - blend_ndcg)

    x = np.vstack(features).astype(np.float32)
    models = {}
    stats: dict[str, Any] = {"n_train": int(x.shape[0])}
    for action in SWITCH_ACTIONS:
        y = np.array(targets[action], dtype=np.float32)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x, y)
        models[action] = model
        stats[f"{action}_target_mean"] = float(np.mean(y))
        stats[f"{action}_target_positive_fraction"] = float(np.mean(y > 0.0))
    return models, stats


def select_threshold(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
    models: dict[str, Any],
    thresholds: list[float],
) -> tuple[float, float]:
    best_threshold = thresholds[0]
    best_metrics = {"NDCG@10": -1.0, "Recall@10": -1.0, "MRR": -1.0}
    for threshold in thresholds:
        result = evaluate_gain_router(cache, weights, missing_score, models, threshold)
        metrics = result["gain_router"]
        candidate_key = (metrics["NDCG@10"], metrics["Recall@10"], metrics["MRR"], threshold)
        best_key = (best_metrics["NDCG@10"], best_metrics["Recall@10"], best_metrics["MRR"], best_threshold)
        if candidate_key > best_key:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics["NDCG@10"]


def evaluate_gain_router(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
    models: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    metric_lists = {
        "gain_router": {metric: [] for metric in METRIC_KEYS},
        "global_blend": {metric: [] for metric in METRIC_KEYS},
        "oracle_all": {metric: [] for metric in METRIC_KEYS},
        **{expert: {metric: [] for metric in METRIC_KEYS} for expert in EXPERT_KEYS},
    }
    action_counts: Counter[str] = Counter()
    oracle_counts: Counter[str] = Counter()

    for user_idx in range(len(cache["users"])):
        gt_items = ground_truth_items(cache, user_idx)
        action_metrics = metrics_by_action(cache, user_idx, weights, missing_score, gt_items)
        selected_action = predict_gain_action(cache, user_idx, models, threshold)
        oracle_action = choose_best(ACTIONS, action_metrics)
        action_counts[selected_action] += 1
        oracle_counts[oracle_action] += 1

        for metric in METRIC_KEYS:
            metric_lists["gain_router"][metric].append(action_metrics[selected_action][metric])
            metric_lists["global_blend"][metric].append(action_metrics["global_blend"][metric])
            metric_lists["oracle_all"][metric].append(action_metrics[oracle_action][metric])
            for expert in EXPERT_KEYS:
                metric_lists[expert][metric].append(action_metrics[expert][metric])

    mean_metrics = {
        method: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for method, metrics in metric_lists.items()
    }
    best_fixed = choose_best(["m7", "r1", "r1plus"], mean_metrics)
    mean_metrics["best_fixed"] = mean_metrics[best_fixed]
    return {
        **mean_metrics,
        "best_fixed_name": best_fixed,
        "action_counts": action_counts,
        "oracle_counts": oracle_counts,
        "n_users": len(cache["users"]),
    }


def predict_gain_action(cache: dict[str, np.ndarray], user_idx: int, models: dict[str, Any], threshold: float) -> str:
    x = user_features(cache, user_idx).reshape(1, -1)
    predicted = {action: float(model.predict(x)[0]) for action, model in models.items()}
    best_action, best_gain = max(predicted.items(), key=lambda item: item[1])
    if best_gain > threshold:
        return best_action
    return "global_blend"


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
    train_stats: dict[str, Any],
    threshold: float,
    val_ndcg: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "n_train": train_stats["n_train"],
        "weights": json.dumps(weights),
        "threshold": threshold,
        "val_gain_router_NDCG@10": val_ndcg,
        "best_fixed": result["best_fixed_name"],
        "best_fixed_NDCG@10": result["best_fixed"]["NDCG@10"],
        "global_blend_NDCG@10": result["global_blend"]["NDCG@10"],
        "gain_router_NDCG@10": result["gain_router"]["NDCG@10"],
        "oracle_all_NDCG@10": result["oracle_all"]["NDCG@10"],
        "router_delta_vs_blend": result["gain_router"]["NDCG@10"] - result["global_blend"]["NDCG@10"],
        "router_delta_vs_best_fixed": result["gain_router"]["NDCG@10"] - result["best_fixed"]["NDCG@10"],
        "oracle_gap_vs_blend": result["oracle_all"]["NDCG@10"] - result["global_blend"]["NDCG@10"],
        "oracle_gap_recovered": recovered(
            result["gain_router"]["NDCG@10"],
            result["global_blend"]["NDCG@10"],
            result["oracle_all"]["NDCG@10"],
        ),
        "gain_router_Recall@10": result["gain_router"]["Recall@10"],
        "global_blend_Recall@10": result["global_blend"]["Recall@10"],
        "gain_router_MRR": result["gain_router"]["MRR"],
        "global_blend_MRR": result["global_blend"]["MRR"],
        **train_stats,
    }


def recovered(router: float, blend: float, oracle: float) -> float:
    gap = oracle - blend
    if gap <= 0:
        return 0.0
    return (router - blend) / gap


def action_count_rows(dataset: dict[str, Any], seed: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source, counter_key in [("router", "action_counts"), ("oracle", "oracle_counts")]:
        total = sum(result[counter_key].values())
        for action in ACTIONS:
            count = result[counter_key].get(action, 0)
            rows.append(
                {
                    "dataset": dataset["name"],
                    "dataset_slug": dataset["slug"],
                    "seed": seed,
                    "source": source,
                    "action": action,
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
                "mean_gain_router_NDCG@10": mean(items, "gain_router_NDCG@10"),
                "mean_oracle_all_NDCG@10": mean(items, "oracle_all_NDCG@10"),
                "mean_router_delta_vs_blend": mean(items, "router_delta_vs_blend"),
                "mean_router_delta_vs_best_fixed": mean(items, "router_delta_vs_best_fixed"),
                "mean_oracle_gap_vs_blend": mean(items, "oracle_gap_vs_blend"),
                "mean_oracle_gap_recovered": mean(items, "oracle_gap_recovered"),
                "mean_gain_router_Recall@10": mean(items, "gain_router_Recall@10"),
                "mean_global_blend_Recall@10": mean(items, "global_blend_Recall@10"),
                "mean_gain_router_MRR": mean(items, "gain_router_MRR"),
                "mean_global_blend_MRR": mean(items, "global_blend_MRR"),
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
        "# V3.1 Pairwise Gain Router Results",
        "",
        "| Dataset | Best Fixed | V2.2 Blend | V3.1 Router | Oracle All | Delta vs Blend | Oracle Gap Recovered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_best_fixed_NDCG@10']:.6f} | "
            f"{row['mean_global_blend_NDCG@10']:.6f} | "
            f"{row['mean_gain_router_NDCG@10']:.6f} | "
            f"{row['mean_oracle_all_NDCG@10']:.6f} | "
            f"{row['mean_router_delta_vs_blend']:.6f} | "
            f"{row['mean_oracle_gap_recovered']:.4f} |"
        )
    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Threshold | V2.2 Blend | V3.1 Router | Delta vs Blend | Oracle Gap Recovered |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['threshold']:.4f} | "
            f"{row['global_blend_NDCG@10']:.6f} | "
            f"{row['gain_router_NDCG@10']:.6f} | "
            f"{row['router_delta_vs_blend']:.6f} | "
            f"{row['oracle_gap_recovered']:.4f} |"
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
