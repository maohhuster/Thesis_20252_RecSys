from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_v2_router import EXPERT_KEYS, load_cache


METRIC_KEYS = ["NDCG@10", "Recall@10", "HR@10", "MRR"]
RANK_MISSING = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V2.2 global blend gains by user/item/expert segments.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_2_segment_analysis.json"))
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
    per_seed_rows: list[dict[str, Any]] = []

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
            segmenters = build_segmenters(cache)
            per_user = evaluate_users(
                cache,
                weights,
                segmenters,
                missing_score=float(config["model"]["missing_score"]),
            )
            per_seed_rows.extend(aggregate_seed_segments(dataset, seed, per_user))
            print(f"  seed={seed}: analyzed {len(per_user)} users", flush=True)

    summary_rows = summarize(per_seed_rows)
    write_csv(output_dir / "segment_by_seed.csv", per_seed_rows)
    write_csv(output_dir / "segment_summary.csv", summary_rows)
    write_markdown(output_dir / "V2_2_Segment_Analysis.md", summary_rows)
    print(f"Wrote segment outputs to {output_dir}", flush=True)


def build_segmenters(cache: dict[str, np.ndarray]) -> dict[str, Any]:
    user_history = cache["user_history_len"].astype(np.float32)
    item_degree = cache["item_degree"].astype(np.float32)
    return {
        "user_activity": quantile_segmenter(user_history, labels=("sparse", "medium", "dense")),
        "target_popularity": quantile_segmenter(item_degree, labels=("cold", "warm", "hot")),
    }


def quantile_segmenter(values: np.ndarray, labels: tuple[str, str, str]):
    finite = values[np.isfinite(values)]
    q1, q2 = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])

    def segment(value: float) -> str:
        if value <= q1:
            return labels[0]
        if value <= q2:
            return labels[1]
        return labels[2]

    return segment


def evaluate_users(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    segmenters: dict[str, Any],
    missing_score: float,
) -> list[dict[str, Any]]:
    rows = []
    n_users = len(cache["users"])
    for user_idx in range(n_users):
        gt_items = ground_truth_items(cache, user_idx)
        if len(gt_items) == 0:
            continue

        ranked_by_method = {
            "global_blend": rank_blend(cache, user_idx, weights, missing_score),
            "m7": cache["m7_top_items"][user_idx],
            "r1": cache["r1_top_items"][user_idx],
            "r1plus": cache["r1plus_top_items"][user_idx],
        }
        metrics = {method: user_metrics(ranked, gt_items) for method, ranked in ranked_by_method.items()}
        best_expert = max(["m7", "r1", "r1plus"], key=lambda method: metric_tuple(metrics[method]))
        segments = user_segments(cache, user_idx, gt_items, segmenters)
        rows.append(
            {
                "user_idx": user_idx,
                "user_id": int(cache["users"][user_idx]),
                "best_expert": best_expert,
                "segments": segments,
                "metrics": metrics,
            }
        )
    return rows


def rank_blend(
    cache: dict[str, np.ndarray],
    user_idx: int,
    weights: tuple[float, float, float],
    missing_score: float,
) -> np.ndarray:
    candidates = candidate_union(cache, user_idx)
    candidate_to_row = {int(item): row for row, item in enumerate(candidates)}
    score_columns = []
    for expert_idx, expert in enumerate(EXPERT_KEYS):
        expert_scores = np.full(len(candidates), missing_score, dtype=np.float32)
        for item, score in zip(cache[f"{expert}_top_items"][user_idx], cache[f"{expert}_top_scores"][user_idx]):
            row = candidate_to_row[int(item)]
            expert_scores[row] = float(score)
        score_columns.append(expert_scores * float(weights[expert_idx]))
    scores = np.sum(np.stack(score_columns, axis=1), axis=1)
    return candidates[np.argsort(-scores)]


def candidate_union(cache: dict[str, np.ndarray], user_idx: int) -> np.ndarray:
    return np.unique(
        np.concatenate([cache[f"{expert}_top_items"][user_idx] for expert in EXPERT_KEYS])
    ).astype(np.int32)


def user_segments(
    cache: dict[str, np.ndarray],
    user_idx: int,
    gt_items: np.ndarray,
    segmenters: dict[str, Any],
) -> dict[str, str]:
    mean_gt_degree = float(np.mean(cache["item_degree"][gt_items])) if len(gt_items) else 0.0
    return {
        "user_activity": segmenters["user_activity"](float(cache["user_history_len"][user_idx])),
        "target_popularity": segmenters["target_popularity"](mean_gt_degree),
        "expert_disagreement": disagreement_segment(expert_disagreement(cache, user_idx)),
    }


def expert_disagreement(cache: dict[str, np.ndarray], user_idx: int) -> float:
    candidates = candidate_union(cache, user_idx)
    candidate_to_row = {int(item): row for row, item in enumerate(candidates)}
    recip_ranks = np.full((len(candidates), len(EXPERT_KEYS)), RANK_MISSING, dtype=np.float32)
    for expert_idx, expert in enumerate(EXPERT_KEYS):
        for rank, item in enumerate(cache[f"{expert}_top_items"][user_idx], start=1):
            row = candidate_to_row[int(item)]
            recip_ranks[row, expert_idx] = 1.0 / rank
    return float(np.mean(np.std(recip_ranks, axis=1)))


def disagreement_segment(value: float) -> str:
    if value < 0.02:
        return "low"
    if value < 0.05:
        return "medium"
    return "high"


def user_metrics(ranked_items: np.ndarray, gt_items: np.ndarray) -> dict[str, float]:
    gt = set(int(item) for item in gt_items)
    top10 = [int(item) for item in ranked_items[:10]]
    hits = np.array([1.0 if item in gt else 0.0 for item in top10], dtype=np.float32)
    dcg = float(np.sum(hits / np.log2(np.arange(2, 2 + len(hits)))))
    idcg = ideal_dcg_at_10(len(gt))
    mrr = 0.0
    for rank, item in enumerate(ranked_items, start=1):
        if int(item) in gt:
            mrr = 1.0 / rank
            break
    return {
        "NDCG@10": dcg / idcg if idcg > 0 else 0.0,
        "Recall@10": float(np.sum(hits)) / len(gt) if gt else 0.0,
        "HR@10": 1.0 if np.any(hits) else 0.0,
        "MRR": mrr,
    }


def ideal_dcg_at_10(gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    return float(np.sum(1.0 / np.log2(np.arange(2, 2 + min(gt_count, 10)))))


def metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["NDCG@10"], metrics["Recall@10"], metrics["MRR"])


def ground_truth_items(cache: dict[str, np.ndarray], user_idx: int) -> np.ndarray:
    indptr = cache["gt_indptr"]
    return cache["gt_items"][indptr[user_idx] : indptr[user_idx + 1]]


def aggregate_seed_segments(
    dataset: dict[str, Any],
    seed: int,
    per_user: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_user:
        for segment_type, segment_value in row["segments"].items():
            groups[(segment_type, segment_value)].append(row)
        groups[("all", "all")].append(row)

    return [
        segment_row(dataset, seed, segment_type, segment_value, rows)
        for (segment_type, segment_value), rows in sorted(groups.items())
    ]


def segment_row(
    dataset: dict[str, Any],
    seed: int,
    segment_type: str,
    segment_value: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blend = mean_metric(rows, "global_blend")
    expert_metrics = {expert: mean_metric(rows, expert) for expert in EXPERT_KEYS}
    best_name, best_metrics = max(expert_metrics.items(), key=lambda item: metric_tuple(item[1]))
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "segment_type": segment_type,
        "segment": segment_value,
        "n_users": len(rows),
        "best_expert": best_name,
        "global_blend_NDCG@10": blend["NDCG@10"],
        "best_expert_NDCG@10": best_metrics["NDCG@10"],
        "delta_NDCG@10": blend["NDCG@10"] - best_metrics["NDCG@10"],
        "global_blend_Recall@10": blend["Recall@10"],
        "best_expert_Recall@10": best_metrics["Recall@10"],
        "delta_Recall@10": blend["Recall@10"] - best_metrics["Recall@10"],
        "global_blend_MRR": blend["MRR"],
        "best_expert_MRR": best_metrics["MRR"],
        "delta_MRR": blend["MRR"] - best_metrics["MRR"],
        "m7_NDCG@10": expert_metrics["m7"]["NDCG@10"],
        "r1_NDCG@10": expert_metrics["r1"]["NDCG@10"],
        "r1plus_NDCG@10": expert_metrics["r1plus"]["NDCG@10"],
    }


def mean_metric(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    return {
        key: float(np.mean([row["metrics"][method][key] for row in rows]))
        for key in METRIC_KEYS
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["dataset_slug"], row["segment_type"], row["segment"])].append(row)

    summary = []
    for (dataset, dataset_slug, segment_type, segment), items in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "segment_type": segment_type,
                "segment": segment,
                "n_seeds": len(items),
                "mean_n_users": float(np.mean([item["n_users"] for item in items])),
                "best_expert_by_seed": ";".join(item["best_expert"] for item in items),
                "mean_global_blend_NDCG@10": float(np.mean([item["global_blend_NDCG@10"] for item in items])),
                "mean_best_expert_NDCG@10": float(np.mean([item["best_expert_NDCG@10"] for item in items])),
                "mean_delta_NDCG@10": float(np.mean([item["delta_NDCG@10"] for item in items])),
                "mean_delta_Recall@10": float(np.mean([item["delta_Recall@10"] for item in items])),
                "mean_delta_MRR": float(np.mean([item["delta_MRR"] for item in items])),
            }
        )
    return summary


def read_selected_weights(path: Path) -> dict[str, tuple[float, float, float]]:
    weights: dict[str, tuple[float, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = ast.literal_eval(row["weights"])
            weights[row["dataset_slug"]] = tuple(float(value) for value in parsed)  # type: ignore[assignment]
    return weights


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V2.2 Segment Analysis",
        "",
        "| Dataset | Segment Type | Segment | Users | Blend NDCG@10 | Best Expert NDCG@10 | Delta | Best Experts by Seed |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["segment_type"] == "all":
            continue
        lines.append(
            f"| {row['dataset']} | {row['segment_type']} | {row['segment']} | "
            f"{row['mean_n_users']:.1f} | {row['mean_global_blend_NDCG@10']:.6f} | "
            f"{row['mean_best_expert_NDCG@10']:.6f} | {row['mean_delta_NDCG@10']:.6f} | "
            f"`{row['best_expert_by_seed']}` |"
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
