from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_v2_1_refined_router import best_expert, candidate_weights, metric_tuple
from train_v2_router import (
    EXPERT_KEYS,
    candidate_union,
    empty_metric_sums,
    evaluate_baselines,
    ground_truth_items,
    ideal_dcg_at_10,
    load_cache,
    score_columns_for_candidates,
)


MetricRow = tuple[np.ndarray, np.ndarray, int, float, dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 segment-level blend experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/v4_segment_level_blend.json"))
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
    strategies = [str(name) for name in config["segment_strategies"]]
    min_segment_users = int(config["min_val_users_per_segment"])
    rows: list[dict[str, Any]] = []
    segment_weight_rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights_grid = candidate_weights(dataset["grid"])
        print(f"=== {dataset['name']} ({len(weights_grid)} weights) ===", flush=True)
        payloads = load_dataset_payloads(
            cache_dir=cache_dir,
            dataset_slug=dataset["slug"],
            top_k=int(config["top_k"]),
            seeds=seeds,
            missing_score=float(config["model"]["missing_score"]),
        )
        if not payloads:
            print("  no cache files found, skipping", flush=True)
            continue

        all_val_rows = [row for payload in payloads for row in payload["val_rows"]]
        global_weights, global_val_metrics = select_weight(all_val_rows, weights_grid)
        print(
            f"  global weights={global_weights} val_NDCG@10={global_val_metrics['NDCG@10']:.6f}",
            flush=True,
        )

        for strategy in strategies:
            segment_weights, segment_counts = select_segment_weights(
                rows=all_val_rows,
                strategy=strategy,
                weights_grid=weights_grid,
                global_weights=global_weights,
                min_segment_users=min_segment_users,
            )
            segment_weight_rows.extend(
                segment_weight_csv_rows(dataset, strategy, segment_weights, segment_counts, global_weights)
            )

            for payload in payloads:
                global_test_metrics = evaluate_weight(payload["test_rows"], global_weights)
                segmented_test_metrics = evaluate_segmented(
                    payload["test_rows"],
                    strategy=strategy,
                    segment_weights=segment_weights,
                    global_weights=global_weights,
                )
                test_best_name, test_best_metrics = best_expert(payload["test_baselines"])
                rows.append(
                    {
                        "dataset": dataset["name"],
                        "dataset_slug": dataset["slug"],
                        "seed": payload["seed"],
                        "method": f"v4_{strategy}",
                        "strategy": strategy,
                        "global_weights": json.dumps(global_weights),
                        "n_segment_weights": len(segment_weights),
                        "selection_mean_val_NDCG@10": global_val_metrics["NDCG@10"],
                        "test_NDCG@10": segmented_test_metrics["NDCG@10"],
                        "test_Recall@10": segmented_test_metrics["Recall@10"],
                        "test_MRR": segmented_test_metrics["MRR"],
                        "v2_2_global_NDCG@10": global_test_metrics["NDCG@10"],
                        "delta_vs_v2_2_global": segmented_test_metrics["NDCG@10"] - global_test_metrics["NDCG@10"],
                        "test_best_expert": test_best_name,
                        "test_best_expert_NDCG@10": test_best_metrics["NDCG@10"],
                        "delta_vs_test_best_expert": segmented_test_metrics["NDCG@10"]
                        - test_best_metrics["NDCG@10"],
                        "m7_NDCG@10": payload["test_baselines"]["m7"]["NDCG@10"],
                        "r1_NDCG@10": payload["test_baselines"]["r1"]["NDCG@10"],
                        "r1plus_NDCG@10": payload["test_baselines"]["r1plus"]["NDCG@10"],
                    }
                )
            print(f"  {strategy}: {len(segment_weights)} segment weights", flush=True)

    summary_rows = summarize(rows)
    write_csv(output_dir / "v4_segment_blend_by_seed.csv", rows)
    write_csv(output_dir / "v4_segment_blend_summary.csv", summary_rows)
    write_csv(output_dir / "v4_segment_weights.csv", segment_weight_rows)
    write_markdown(output_dir / "V4_Segment_Level_Blend.md", summary_rows, segment_weight_rows, rows)
    print(f"Wrote V4 outputs to {output_dir}", flush=True)


def load_dataset_payloads(
    cache_dir: Path,
    dataset_slug: str,
    top_k: int,
    seeds: list[int],
    missing_score: float,
) -> list[dict[str, Any]]:
    val_caches = []
    test_caches = []
    cache_seeds = []
    for seed in seeds:
        val_path = cache_dir / dataset_slug / f"seed-{seed}" / f"val_top{top_k}.npz"
        test_path = cache_dir / dataset_slug / f"seed-{seed}" / f"test_top{top_k}.npz"
        if not val_path.exists() or not test_path.exists():
            print(f"  seed={seed}: missing cache, skipping", flush=True)
            continue
        val_caches.append(load_cache(val_path))
        test_caches.append(load_cache(test_path))
        cache_seeds.append(seed)

    if not val_caches:
        return []

    thresholds = fit_segment_thresholds(val_caches)
    payloads = []
    for seed, val_cache, test_cache in zip(cache_seeds, val_caches, test_caches):
        payloads.append(
            {
                "seed": seed,
                "val_rows": prepare_segment_rows(val_cache, thresholds, missing_score),
                "test_rows": prepare_segment_rows(test_cache, thresholds, missing_score),
                "test_baselines": evaluate_baselines(test_cache),
            }
        )
    return payloads


def fit_segment_thresholds(caches: list[dict[str, np.ndarray]]) -> dict[str, tuple[float, float]]:
    user_history = np.concatenate([cache["user_history_len"].astype(np.float32) for cache in caches])
    disagreement_values = []
    for cache in caches:
        for user_idx in range(len(cache["users"])):
            disagreement_values.append(expert_disagreement(cache, user_idx))
    disagreement = np.array(disagreement_values, dtype=np.float32)
    return {
        "user_activity": quantile_pair(user_history),
        "expert_disagreement": quantile_pair(disagreement),
    }


def quantile_pair(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (0.0, 0.0)
    q1, q2 = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    return (float(q1), float(q2))


def prepare_segment_rows(
    cache: dict[str, np.ndarray],
    thresholds: dict[str, tuple[float, float]],
    missing_score: float,
) -> list[MetricRow]:
    rows = []
    for user_idx in range(len(cache["users"])):
        candidates = candidate_union(cache, user_idx)
        scores = score_columns_for_candidates(cache, user_idx, candidates, missing_score)
        gt_items = ground_truth_items(cache, user_idx)
        gt_count = int(len(gt_items))
        segments = {
            "user_activity": tertile_label(
                float(cache["user_history_len"][user_idx]),
                thresholds["user_activity"],
                ("sparse", "medium", "dense"),
            ),
            "expert_disagreement": tertile_label(
                expert_disagreement(cache, user_idx),
                thresholds["expert_disagreement"],
                ("low", "medium", "high"),
            ),
        }
        segments["user_activity_x_expert_disagreement"] = (
            f"{segments['user_activity']}|{segments['expert_disagreement']}"
        )
        rows.append((scores, np.isin(candidates, gt_items), gt_count, ideal_dcg_at_10(gt_count), segments))
    return rows


def tertile_label(value: float, thresholds: tuple[float, float], labels: tuple[str, str, str]) -> str:
    q1, q2 = thresholds
    if value <= q1:
        return labels[0]
    if value <= q2:
        return labels[1]
    return labels[2]


def expert_disagreement(cache: dict[str, np.ndarray], user_idx: int) -> float:
    candidates = candidate_union(cache, user_idx)
    candidate_to_row = {int(item): row for row, item in enumerate(candidates)}
    recip_ranks = np.zeros((len(candidates), len(EXPERT_KEYS)), dtype=np.float32)
    for expert_idx, expert in enumerate(EXPERT_KEYS):
        for rank, item in enumerate(cache[f"{expert}_top_items"][user_idx], start=1):
            row = candidate_to_row.get(int(item))
            if row is not None:
                recip_ranks[row, expert_idx] = 1.0 / rank
    return float(np.mean(np.std(recip_ranks, axis=1))) if len(candidates) else 0.0


def select_segment_weights(
    rows: list[MetricRow],
    strategy: str,
    weights_grid: list[tuple[float, float, float]],
    global_weights: tuple[float, float, float],
    min_segment_users: int,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, int]]:
    grouped: dict[str, list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[row[4][strategy]].append(row)

    segment_weights = {}
    segment_counts = {}
    for segment, segment_rows in sorted(grouped.items()):
        segment_counts[segment] = len(segment_rows)
        if len(segment_rows) < min_segment_users:
            segment_weights[segment] = global_weights
            continue
        segment_weights[segment], _ = select_weight(segment_rows, weights_grid)
    return segment_weights, segment_counts


def select_weight(
    rows: list[MetricRow],
    weights_grid: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], dict[str, float]]:
    best_weights = weights_grid[0]
    best_metrics = {"NDCG@10": -1.0, "Recall@10": -1.0, "MRR": -1.0}
    for weights in weights_grid:
        metrics = evaluate_weight(rows, weights)
        if metric_tuple(metrics) > metric_tuple(best_metrics):
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def evaluate_weight(rows: list[MetricRow], weights: tuple[float, float, float]) -> dict[str, float]:
    weight_arr = np.array(weights, dtype=np.float32)
    return evaluate_with_weight_getter(rows, lambda _segments: weight_arr)


def evaluate_segmented(
    rows: list[MetricRow],
    strategy: str,
    segment_weights: dict[str, tuple[float, float, float]],
    global_weights: tuple[float, float, float],
) -> dict[str, float]:
    fallback = np.array(global_weights, dtype=np.float32)
    weight_arrays = {
        segment: np.array(weights, dtype=np.float32)
        for segment, weights in segment_weights.items()
    }

    def get_weight(segments: dict[str, str]) -> np.ndarray:
        return weight_arrays.get(segments[strategy], fallback)

    return evaluate_with_weight_getter(rows, get_weight)


def evaluate_with_weight_getter(rows: list[MetricRow], get_weight) -> dict[str, float]:
    sums = empty_metric_sums()
    n_users = len(rows)
    for scores_by_expert, gt_mask, gt_count, idcg, segments in rows:
        if gt_count == 0:
            continue
        scores = scores_by_expert @ get_weight(segments)
        hit_mask = gt_mask[np.argsort(-scores)]
        top_hits = hit_mask[:10].astype(np.float32)
        sums["HR@10"] += 1.0 if np.any(top_hits) else 0.0
        sums["Recall@10"] += float(np.sum(top_hits)) / gt_count
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(top_hits)))
        sums["NDCG@10"] += float(np.sum(top_hits * discounts) / idcg) if idcg > 0 else 0.0
        first_hit = np.flatnonzero(hit_mask)
        if len(first_hit) > 0:
            sums["MRR"] += 1.0 / float(first_hit[0] + 1)
    return {key: value / n_users for key, value in sums.items()}


def segment_weight_csv_rows(
    dataset: dict[str, Any],
    strategy: str,
    segment_weights: dict[str, tuple[float, float, float]],
    segment_counts: dict[str, int],
    global_weights: tuple[float, float, float],
) -> list[dict[str, Any]]:
    rows = []
    for segment, weights in sorted(segment_weights.items()):
        rows.append(
            {
                "dataset": dataset["name"],
                "dataset_slug": dataset["slug"],
                "strategy": strategy,
                "segment": segment,
                "n_val_users": segment_counts[segment],
                "weights": json.dumps(weights),
                "uses_global_fallback": weights == global_weights,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["dataset_slug"], row["method"]), []).append(row)

    summary = []
    for (dataset, dataset_slug, method), items in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "method": method,
                "n_seeds": len(items),
                "mean_NDCG@10": float(np.mean([item["test_NDCG@10"] for item in items])),
                "std_NDCG@10": float(np.std([item["test_NDCG@10"] for item in items])),
                "mean_v2_2_global_NDCG@10": float(np.mean([item["v2_2_global_NDCG@10"] for item in items])),
                "mean_delta_vs_v2_2_global": float(np.mean([item["delta_vs_v2_2_global"] for item in items])),
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
    summary_rows: list[dict[str, Any]],
    segment_weight_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# V4 Segment-Level Blend",
        "",
        "V4 learns blend weights per deployable segment and compares them with the",
        "V2.2 dataset-level global blend. It intentionally avoids target-item",
        "segments because target ground-truth is unavailable at deployment time.",
        "",
        "## Summary",
        "",
        "| Dataset | Method | Seeds | NDCG@10 | V2.2 Global | Delta vs V2.2 | Best Expert | Delta vs Best |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | `{row['method']}` | {row['n_seeds']} | "
            f"{row['mean_NDCG@10']:.6f} | {row['mean_v2_2_global_NDCG@10']:.6f} | "
            f"{row['mean_delta_vs_v2_2_global']:.6f} | "
            f"{row['mean_test_best_expert_NDCG@10']:.6f} | "
            f"{row['mean_delta_vs_test_best_expert']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Segment Weights",
            "",
            "| Dataset | Strategy | Segment | Val Users | Weights | Fallback |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in segment_weight_rows:
        lines.append(
            f"| {row['dataset']} | `{row['strategy']}` | `{row['segment']}` | "
            f"{row['n_val_users']} | `{row['weights']}` | {row['uses_global_fallback']} |"
        )

    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Method | NDCG@10 | V2.2 Global | Delta vs V2.2 | Test Best | Delta vs Best |",
            "|---|---:|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | `{row['method']}` | "
            f"{row['test_NDCG@10']:.6f} | {row['v2_2_global_NDCG@10']:.6f} | "
            f"{row['delta_vs_v2_2_global']:.6f} | "
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
