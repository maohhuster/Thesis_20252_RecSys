from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_v2_2_segments import (
    METRIC_KEYS,
    candidate_union,
    expert_disagreement,
    ground_truth_items,
    rank_blend,
    read_selected_weights,
    user_metrics,
)
from run_v2_4_oracle_gap import choose_best
from train_v2_router import EXPERT_KEYS, load_cache


ACTIONS = ["global_blend", *EXPERT_KEYS]
FEATURE_NAMES = [
    "log_user_history_len",
    "candidate_log_degree_mean",
    "candidate_log_degree_std",
    "candidate_log_degree_max",
    "expert_disagreement",
    "m7_r1_top10_jaccard",
    "m7_r1plus_top10_jaccard",
    "r1_r1plus_top10_jaccard",
    "m7_r1_top100_jaccard",
    "m7_r1plus_top100_jaccard",
    "r1_r1plus_top100_jaccard",
    "m7_top1_score",
    "r1_top1_score",
    "r1plus_top1_score",
    "m7_top10_mean_score",
    "r1_top10_mean_score",
    "r1plus_top10_mean_score",
    "m7_minus_r1_top1",
    "m7_minus_r1plus_top1",
    "r1_minus_r1plus_top1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3 per-user action router.")
    parser.add_argument("--config", type=Path, default=Path("configs/v3_action_router.json"))
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
    coef_rows: list[dict[str, Any]] = []

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
            x_train, y_train, train_counts = build_training_data(
                cache=val_cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
                exclude_ties=bool(config["model"]["exclude_tied_oracle_labels"]),
                min_label_ndcg=float(config["model"]["min_label_NDCG@10"]),
            )
            if len(set(y_train.tolist())) < 2:
                print(f"  seed={seed}: not enough action classes, fallback to global_blend", flush=True)
                model = None
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(config["model"]["C"]),
                        class_weight=config["model"].get("class_weight"),
                        max_iter=int(config["model"]["max_iter"]),
                        random_state=seed,
                    ),
                )
                model.fit(x_train, y_train)
                coef_rows.extend(coefficient_rows(model, dataset, seed))

            threshold, val_router_ndcg = select_confidence_threshold(
                cache=val_cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
                model=model,
                thresholds=[float(value) for value in config["model"]["confidence_thresholds"]],
            )
            result = evaluate_router(
                cache=test_cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
                model=model,
                confidence_threshold=threshold,
            )
            by_seed_rows.append(
                seed_row(dataset, seed, weights, result, train_counts, len(y_train), threshold, val_router_ndcg)
            )
            action_rows.extend(action_count_rows(dataset, seed, result))
            print(
                f"  seed={seed}: train={len(y_train)} threshold={threshold:.2f} "
                f"router={result['action_router']['NDCG@10']:.6f} "
                f"blend={result['global_blend']['NDCG@10']:.6f} "
                f"delta={result['action_router']['NDCG@10'] - result['global_blend']['NDCG@10']:.6f}",
                flush=True,
            )

    summary_rows = summarize(by_seed_rows)
    write_csv(output_dir / "v3_action_router_by_seed.csv", by_seed_rows)
    write_csv(output_dir / "v3_action_router_summary.csv", summary_rows)
    write_csv(output_dir / "v3_action_router_action_counts.csv", action_rows)
    write_csv(output_dir / "v3_action_router_coefficients.csv", coef_rows)
    write_markdown(output_dir / "V3_Action_Router_Results.md", summary_rows, by_seed_rows)
    print(f"Wrote V3 action-router outputs to {output_dir}", flush=True)


def build_training_data(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
    exclude_ties: bool,
    min_label_ndcg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    features = []
    labels = []
    counts: Counter[str] = Counter()
    for user_idx in range(len(cache["users"])):
        gt_items = ground_truth_items(cache, user_idx)
        action_metrics = metrics_by_action(cache, user_idx, weights, missing_score, gt_items)
        ordered = sorted(ACTIONS, key=lambda action: metric_tuple(action_metrics[action]), reverse=True)
        best = ordered[0]
        second = ordered[1]
        if action_metrics[best]["NDCG@10"] <= min_label_ndcg:
            continue
        if exclude_ties and metric_tuple(action_metrics[best]) == metric_tuple(action_metrics[second]):
            continue
        features.append(user_features(cache, user_idx))
        labels.append(ACTIONS.index(best))
        counts[best] += 1

    if not features:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), dict(counts)
    return np.vstack(features).astype(np.float32), np.array(labels, dtype=np.int64), dict(counts)


def evaluate_router(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
    model,
    confidence_threshold: float,
) -> dict[str, Any]:
    metric_lists = {
        "action_router": {metric: [] for metric in METRIC_KEYS},
        "global_blend": {metric: [] for metric in METRIC_KEYS},
        "oracle_all": {metric: [] for metric in METRIC_KEYS},
        **{expert: {metric: [] for metric in METRIC_KEYS} for expert in EXPERT_KEYS},
    }
    action_counts: Counter[str] = Counter()
    oracle_counts: Counter[str] = Counter()

    for user_idx in range(len(cache["users"])):
        gt_items = ground_truth_items(cache, user_idx)
        action_metrics = metrics_by_action(cache, user_idx, weights, missing_score, gt_items)
        if model is None:
            selected_action = "global_blend"
        else:
            selected_action = predict_guarded_action(model, cache, user_idx, confidence_threshold)
        oracle_action = choose_best(ACTIONS, action_metrics)
        action_counts[selected_action] += 1
        oracle_counts[oracle_action] += 1

        for metric in METRIC_KEYS:
            metric_lists["action_router"][metric].append(action_metrics[selected_action][metric])
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


def select_confidence_threshold(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
    model,
    thresholds: list[float],
) -> tuple[float, float]:
    if model is None:
        metrics = evaluate_router(cache, weights, missing_score, model, confidence_threshold=1.0)
        return 1.0, metrics["action_router"]["NDCG@10"]

    best_threshold = thresholds[0]
    best_metrics = {"NDCG@10": -1.0, "Recall@10": -1.0, "MRR": -1.0}
    for threshold in thresholds:
        metrics = evaluate_router(cache, weights, missing_score, model, confidence_threshold=threshold)
        router_metrics = metrics["action_router"]
        candidate_key = (
            router_metrics["NDCG@10"],
            router_metrics["Recall@10"],
            router_metrics["MRR"],
            threshold,
        )
        best_key = (
            best_metrics["NDCG@10"],
            best_metrics["Recall@10"],
            best_metrics["MRR"],
            best_threshold,
        )
        if candidate_key > best_key:
            best_threshold = threshold
            best_metrics = router_metrics
    return best_threshold, best_metrics["NDCG@10"]


def predict_guarded_action(model, cache: dict[str, np.ndarray], user_idx: int, confidence_threshold: float) -> str:
    probabilities = model.predict_proba(user_features(cache, user_idx).reshape(1, -1))[0]
    best_idx = int(np.argmax(probabilities))
    if float(probabilities[best_idx]) < confidence_threshold:
        return "global_blend"
    return ACTIONS[int(model.classes_[best_idx])]


def metrics_by_action(
    cache: dict[str, np.ndarray],
    user_idx: int,
    weights: tuple[float, float, float],
    missing_score: float,
    gt_items: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "global_blend": user_metrics(rank_blend(cache, user_idx, weights, missing_score), gt_items),
        "m7": user_metrics(cache["m7_top_items"][user_idx], gt_items),
        "r1": user_metrics(cache["r1_top_items"][user_idx], gt_items),
        "r1plus": user_metrics(cache["r1plus_top_items"][user_idx], gt_items),
    }


def user_features(cache: dict[str, np.ndarray], user_idx: int) -> np.ndarray:
    union = candidate_union(cache, user_idx)
    log_degrees = np.log1p(cache["item_degree"][union].astype(np.float32))
    top1_scores = [float(cache[f"{expert}_top_scores"][user_idx, 0]) for expert in EXPERT_KEYS]
    top10_scores = [float(np.mean(cache[f"{expert}_top_scores"][user_idx, :10])) for expert in EXPERT_KEYS]
    m7_top10 = cache["m7_top_items"][user_idx, :10]
    r1_top10 = cache["r1_top_items"][user_idx, :10]
    r1plus_top10 = cache["r1plus_top_items"][user_idx, :10]
    m7_top100 = cache["m7_top_items"][user_idx]
    r1_top100 = cache["r1_top_items"][user_idx]
    r1plus_top100 = cache["r1plus_top_items"][user_idx]
    return np.array(
        [
            np.log1p(float(cache["user_history_len"][user_idx])),
            float(np.mean(log_degrees)),
            float(np.std(log_degrees)),
            float(np.max(log_degrees)),
            expert_disagreement(cache, user_idx),
            jaccard(m7_top10, r1_top10),
            jaccard(m7_top10, r1plus_top10),
            jaccard(r1_top10, r1plus_top10),
            jaccard(m7_top100, r1_top100),
            jaccard(m7_top100, r1plus_top100),
            jaccard(r1_top100, r1plus_top100),
            *top1_scores,
            *top10_scores,
            top1_scores[0] - top1_scores[1],
            top1_scores[0] - top1_scores[2],
            top1_scores[1] - top1_scores[2],
        ],
        dtype=np.float32,
    )


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_set = set(int(item) for item in a)
    b_set = set(int(item) for item in b)
    union = len(a_set | b_set)
    if union == 0:
        return 0.0
    return len(a_set & b_set) / union


def metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["NDCG@10"], metrics["Recall@10"], metrics["MRR"])


def seed_row(
    dataset: dict[str, Any],
    seed: int,
    weights: tuple[float, float, float],
    result: dict[str, Any],
    train_counts: dict[str, int],
    n_train: int,
    confidence_threshold: float,
    val_router_ndcg: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "n_train": n_train,
        "train_label_counts_json": json.dumps(train_counts, sort_keys=True),
        "n_test_users": result["n_users"],
        "weights": json.dumps(weights),
        "confidence_threshold": confidence_threshold,
        "val_action_router_NDCG@10": val_router_ndcg,
        "best_fixed": result["best_fixed_name"],
        "best_fixed_NDCG@10": result["best_fixed"]["NDCG@10"],
        "global_blend_NDCG@10": result["global_blend"]["NDCG@10"],
        "action_router_NDCG@10": result["action_router"]["NDCG@10"],
        "oracle_all_NDCG@10": result["oracle_all"]["NDCG@10"],
        "router_delta_vs_blend": result["action_router"]["NDCG@10"] - result["global_blend"]["NDCG@10"],
        "router_delta_vs_best_fixed": result["action_router"]["NDCG@10"] - result["best_fixed"]["NDCG@10"],
        "oracle_gap_vs_blend": result["oracle_all"]["NDCG@10"] - result["global_blend"]["NDCG@10"],
        "oracle_gap_remaining_after_router": result["oracle_all"]["NDCG@10"] - result["action_router"]["NDCG@10"],
        "oracle_gap_recovered": recovered(
            result["action_router"]["NDCG@10"],
            result["global_blend"]["NDCG@10"],
            result["oracle_all"]["NDCG@10"],
        ),
        "action_router_Recall@10": result["action_router"]["Recall@10"],
        "global_blend_Recall@10": result["global_blend"]["Recall@10"],
        "action_router_MRR": result["action_router"]["MRR"],
        "global_blend_MRR": result["global_blend"]["MRR"],
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


def coefficient_rows(model, dataset: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    classifier = model.named_steps["logisticregression"]
    rows = []
    for class_idx, action_id in enumerate(classifier.classes_):
        action = ACTIONS[int(action_id)]
        for feature, coef in zip(FEATURE_NAMES, classifier.coef_[class_idx]):
            rows.append(
                {
                    "dataset": dataset["name"],
                    "dataset_slug": dataset["slug"],
                    "seed": seed,
                    "action": action,
                    "feature": feature,
                    "coef": float(coef),
                    "abs_coef": float(abs(coef)),
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
                "mean_action_router_NDCG@10": mean(items, "action_router_NDCG@10"),
                "mean_oracle_all_NDCG@10": mean(items, "oracle_all_NDCG@10"),
                "mean_router_delta_vs_blend": mean(items, "router_delta_vs_blend"),
                "mean_router_delta_vs_best_fixed": mean(items, "router_delta_vs_best_fixed"),
                "mean_oracle_gap_vs_blend": mean(items, "oracle_gap_vs_blend"),
                "mean_oracle_gap_recovered": mean(items, "oracle_gap_recovered"),
                "mean_action_router_Recall@10": mean(items, "action_router_Recall@10"),
                "mean_global_blend_Recall@10": mean(items, "global_blend_Recall@10"),
                "mean_action_router_MRR": mean(items, "action_router_MRR"),
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
        "# V3 Action Router Results",
        "",
        "| Dataset | Best Fixed | V2.2 Blend | V3 Router | Oracle All | Delta vs Blend | Oracle Gap Recovered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_best_fixed_NDCG@10']:.6f} | "
            f"{row['mean_global_blend_NDCG@10']:.6f} | "
            f"{row['mean_action_router_NDCG@10']:.6f} | "
            f"{row['mean_oracle_all_NDCG@10']:.6f} | "
            f"{row['mean_router_delta_vs_blend']:.6f} | "
            f"{row['mean_oracle_gap_recovered']:.4f} |"
        )
    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Train Labels | V2.2 Blend | V3 Router | Delta vs Blend | Oracle Gap Recovered |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['n_train']} | "
            f"{row['global_blend_NDCG@10']:.6f} | "
            f"{row['action_router_NDCG@10']:.6f} | "
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
