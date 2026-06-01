from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERT_KEYS = ["m7", "r1", "r1plus"]
FEATURE_NAMES = [
    "log_item_degree",
    "log_user_history_len",
    "m7_score",
    "r1_score",
    "r1plus_score",
    "m7_recip_rank",
    "r1_recip_rank",
    "r1plus_recip_rank",
    "m7_present",
    "r1_present",
    "r1plus_present",
    "m7_minus_r1",
    "m7_minus_r1plus",
    "r1_minus_r1plus",
    "max_score",
    "score_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2 candidate-router baselines from cached expert scores.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_router.json"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config(args.config)
    config = _read_json(config_path)
    cache_dir = _resolve_path(config["candidate_cache_dir"], config_path)
    output_dir = _resolve_path(config["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = set(args.datasets) if args.datasets else None
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}")
            continue

        print(f"=== {dataset['name']} ===", flush=True)
        for seed in seeds:
            train_path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"val_top{config['top_k']}.npz"
            test_path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"test_top{config['top_k']}.npz"
            if not train_path.exists() or not test_path.exists():
                print(f"  seed={seed}: missing cache, skipping", flush=True)
                continue

            train_cache = load_cache(train_path)
            test_cache = load_cache(test_path)
            train_baselines = evaluate_baselines(train_cache)
            test_baselines = evaluate_baselines(test_cache)
            best_baseline_name, best_baseline = max(
                test_baselines.items(),
                key=lambda item: item[1]["NDCG@10"],
            )

            train_blend_rows = prepare_blend_rows(
                train_cache,
                missing_score=float(config["model"]["missing_score"]),
            )
            test_blend_rows = prepare_blend_rows(
                test_cache,
                missing_score=float(config["model"]["missing_score"]),
            )
            best_weights, blend_val_metrics = select_blend_weights(
                train_blend_rows,
                step=float(config["model"]["blend_grid_step"]),
            )
            blend_test_metrics = evaluate_blend_cache(
                test_blend_rows,
                weights=best_weights,
            )
            rows.append(
                result_row(
                    dataset=dataset,
                    seed=seed,
                    method="blend_grid",
                    train_rows=0,
                    train_positive_rate=0.0,
                    val_metrics=blend_val_metrics,
                    test_metrics=blend_test_metrics,
                    best_baseline_name=best_baseline_name,
                    best_baseline=best_baseline,
                    baselines=test_baselines,
                    weights=best_weights,
                )
            )
            print(
                f"  seed={seed}: blend weights={best_weights} "
                f"test NDCG@10={blend_test_metrics['NDCG@10']:.6f}, "
                f"best={best_baseline_name} {best_baseline['NDCG@10']:.6f}",
                flush=True,
            )

            if not bool(config["model"].get("run_logistic", False)):
                continue

            print(f"  seed={seed}: building logistic train features", flush=True)
            rng = np.random.default_rng(seed)
            x_train, y_train = build_training_matrix(
                train_cache,
                missing_score=float(config["model"]["missing_score"]),
                max_negatives_per_user=int(config["model"]["max_train_negatives_per_user"]),
                rng=rng,
            )
            if int(y_train.sum()) == 0:
                print(f"  seed={seed}: no positive cached candidates, skipping", flush=True)
                continue

            model = make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=float(config["model"]["alpha"]),
                    max_iter=int(config["model"]["max_iter"]),
                    tol=float(config["model"]["tol"]),
                    class_weight=config["model"].get("class_weight"),
                    random_state=seed,
                ),
            )
            model.fit(x_train, y_train)

            train_metrics = evaluate_logistic_cache(
                model,
                train_cache,
                missing_score=float(config["model"]["missing_score"]),
            )
            test_metrics = evaluate_logistic_cache(
                model,
                test_cache,
                missing_score=float(config["model"]["missing_score"]),
            )
            row = result_row(
                dataset=dataset,
                seed=seed,
                method="logistic",
                train_rows=int(x_train.shape[0]),
                train_positive_rate=float(y_train.mean()),
                val_metrics=train_metrics,
                test_metrics=test_metrics,
                best_baseline_name=best_baseline_name,
                best_baseline=best_baseline,
                baselines=test_baselines,
                weights=None,
            )
            rows.append(row)
            print(
                f"  seed={seed}: logistic test NDCG@10={row['test_NDCG@10']:.6f}, "
                f"best={best_baseline_name} {row['best_baseline_NDCG@10']:.6f}, "
                f"delta={row['delta_vs_best_baseline']:.6f}",
                flush=True,
            )

            feature_rows.extend(feature_importance_rows(model, dataset["name"], dataset["slug"], seed))

    summary_rows = summarize(rows)
    _write_csv(output_dir / "v2_router_by_seed.csv", rows)
    _write_csv(output_dir / "v2_router_summary.csv", summary_rows)
    _write_csv(output_dir / "v2_router_coefficients.csv", feature_rows)
    _write_markdown(output_dir / "V2_Router_Results.md", summary_rows, rows)
    print(f"Wrote V2 router outputs to {output_dir}", flush=True)


def build_training_matrix(
    cache: dict[str, np.ndarray],
    missing_score: float,
    max_negatives_per_user: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    n_users = len(cache["users"])

    for user_idx in range(n_users):
        candidates = candidate_union(cache, user_idx)
        labels = labels_for_candidates(cache, user_idx, candidates)
        positive_idx = np.flatnonzero(labels == 1)
        negative_idx = np.flatnonzero(labels == 0)
        if max_negatives_per_user > 0 and len(negative_idx) > max_negatives_per_user:
            negative_idx = rng.choice(negative_idx, size=max_negatives_per_user, replace=False)
        keep_idx = np.sort(np.concatenate([positive_idx, negative_idx]))
        if len(keep_idx) == 0:
            continue
        kept = candidates[keep_idx]
        feature_blocks.append(features_for_candidates(cache, user_idx, kept, missing_score))
        label_blocks.append(labels[keep_idx])

    if not feature_blocks:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int8)
    return np.vstack(feature_blocks).astype(np.float32), np.concatenate(label_blocks).astype(np.int8)


def result_row(
    dataset: dict[str, Any],
    seed: int,
    method: str,
    train_rows: int,
    train_positive_rate: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_baseline_name: str,
    best_baseline: dict[str, float],
    baselines: dict[str, dict[str, float]],
    weights: tuple[float, float, float] | None,
) -> dict[str, Any]:
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "method": method,
        "weights": json.dumps(weights) if weights is not None else "",
        "train_rows": train_rows,
        "train_positive_rate": train_positive_rate,
        "val_NDCG@10": val_metrics["NDCG@10"],
        "test_NDCG@10": test_metrics["NDCG@10"],
        "test_Recall@10": test_metrics["Recall@10"],
        "test_MRR": test_metrics["MRR"],
        "best_baseline": best_baseline_name,
        "best_baseline_NDCG@10": best_baseline["NDCG@10"],
        "delta_vs_best_baseline": test_metrics["NDCG@10"] - best_baseline["NDCG@10"],
        "m7_NDCG@10": baselines["m7"]["NDCG@10"],
        "r1_NDCG@10": baselines["r1"]["NDCG@10"],
        "r1plus_NDCG@10": baselines["r1plus"]["NDCG@10"],
    }


def evaluate_logistic_cache(
    model,
    cache: dict[str, np.ndarray],
    missing_score: float,
) -> dict[str, float]:
    sums = empty_metric_sums()
    n_users = len(cache["users"])
    for user_idx in range(n_users):
        candidates = candidate_union(cache, user_idx)
        if len(candidates) == 0:
            continue
        x = features_for_candidates(cache, user_idx, candidates, missing_score)
        scores = model.predict_proba(x)[:, 1]
        ranked = candidates[np.argsort(-scores)]
        accumulate_metrics(sums, ranked, ground_truth_items(cache, user_idx))
    return {key: value / n_users for key, value in sums.items()}


def select_blend_weights(
    prepared_rows: list[tuple[np.ndarray, np.ndarray, int, float]],
    step: float,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    best_weights = (1.0, 0.0, 0.0)
    best_metrics = {"NDCG@10": -1.0}
    for weights in simplex_weights(step):
        metrics = evaluate_blend_cache(prepared_rows, weights)
        if (metrics["NDCG@10"], metrics["Recall@10"], metrics["MRR"]) > (
            best_metrics.get("NDCG@10", -1.0),
            best_metrics.get("Recall@10", -1.0),
            best_metrics.get("MRR", -1.0),
        ):
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def simplex_weights(step: float) -> list[tuple[float, float, float]]:
    scale = int(round(1.0 / step))
    weights = []
    for a in range(scale + 1):
        for b in range(scale + 1 - a):
            c = scale - a - b
            weights.append((a / scale, b / scale, c / scale))
    return weights


def evaluate_blend_cache(
    prepared_rows: list[tuple[np.ndarray, np.ndarray, int, float]],
    weights: tuple[float, float, float],
) -> dict[str, float]:
    sums = empty_metric_sums()
    n_users = len(prepared_rows)
    weight_arr = np.array(weights, dtype=np.float32)
    for scores_by_expert, gt_mask, gt_count, idcg in prepared_rows:
        if gt_count == 0:
            continue
        scores = scores_by_expert @ weight_arr
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


def prepare_blend_rows(
    cache: dict[str, np.ndarray],
    missing_score: float,
) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    rows = []
    for user_idx in range(len(cache["users"])):
        candidates = candidate_union(cache, user_idx)
        scores = score_columns_for_candidates(cache, user_idx, candidates, missing_score)
        gt_items = ground_truth_items(cache, user_idx)
        gt_count = int(len(gt_items))
        rows.append((scores, np.isin(candidates, gt_items), gt_count, ideal_dcg_at_10(gt_count)))
    return rows


def ideal_dcg_at_10(gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    ranks = np.arange(2, 2 + min(gt_count, 10))
    return float(np.sum(1.0 / np.log2(ranks)))


def evaluate_baselines(cache: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    n_users = len(cache["users"])
    out: dict[str, dict[str, float]] = {}
    for expert in EXPERT_KEYS:
        sums = empty_metric_sums()
        top_items = cache[f"{expert}_top_items"]
        for user_idx in range(n_users):
            accumulate_metrics(sums, top_items[user_idx], ground_truth_items(cache, user_idx))
        out[expert] = {key: value / n_users for key, value in sums.items()}
    return out


def candidate_union(cache: dict[str, np.ndarray], user_idx: int) -> np.ndarray:
    return np.unique(
        np.concatenate([cache[f"{expert}_top_items"][user_idx] for expert in EXPERT_KEYS])
    ).astype(np.int32)


def features_for_candidates(
    cache: dict[str, np.ndarray],
    user_idx: int,
    candidates: np.ndarray,
    missing_score: float,
) -> np.ndarray:
    n = len(candidates)
    item_degree = cache["item_degree"][candidates].astype(np.float32)
    user_history = float(cache["user_history_len"][user_idx])
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = np.log1p(item_degree)
    features[:, 1] = np.log1p(user_history)

    score_cols = []
    rank_cols = []
    present_cols = []
    candidate_to_row = {int(item): row for row, item in enumerate(candidates)}

    for expert in EXPERT_KEYS:
        scores = np.full(n, missing_score, dtype=np.float32)
        recip_rank = np.zeros(n, dtype=np.float32)
        present = np.zeros(n, dtype=np.float32)
        top_items = cache[f"{expert}_top_items"][user_idx]
        top_scores = cache[f"{expert}_top_scores"][user_idx]
        for rank, (item, score) in enumerate(zip(top_items, top_scores), start=1):
            row = candidate_to_row.get(int(item))
            if row is None:
                continue
            scores[row] = float(score)
            recip_rank[row] = 1.0 / rank
            present[row] = 1.0
        score_cols.append(scores)
        rank_cols.append(recip_rank)
        present_cols.append(present)

    features[:, 2:5] = np.stack(score_cols, axis=1)
    features[:, 5:8] = np.stack(rank_cols, axis=1)
    features[:, 8:11] = np.stack(present_cols, axis=1)
    features[:, 11] = features[:, 2] - features[:, 3]
    features[:, 12] = features[:, 2] - features[:, 4]
    features[:, 13] = features[:, 3] - features[:, 4]
    features[:, 14] = np.max(features[:, 2:5], axis=1)
    features[:, 15] = np.std(features[:, 2:5], axis=1)
    return features


def score_columns_for_candidates(
    cache: dict[str, np.ndarray],
    user_idx: int,
    candidates: np.ndarray,
    missing_score: float,
) -> np.ndarray:
    n = len(candidates)
    candidate_to_row = {int(item): row for row, item in enumerate(candidates)}
    columns = []
    for expert in EXPERT_KEYS:
        scores = np.full(n, missing_score, dtype=np.float32)
        top_items = cache[f"{expert}_top_items"][user_idx]
        top_scores = cache[f"{expert}_top_scores"][user_idx]
        for item, score in zip(top_items, top_scores):
            row = candidate_to_row.get(int(item))
            if row is not None:
                scores[row] = float(score)
        columns.append(scores)
    return np.stack(columns, axis=1)


def labels_for_candidates(
    cache: dict[str, np.ndarray],
    user_idx: int,
    candidates: np.ndarray,
) -> np.ndarray:
    gt = set(int(item) for item in ground_truth_items(cache, user_idx))
    return np.array([1 if int(item) in gt else 0 for item in candidates], dtype=np.int8)


def ground_truth_items(cache: dict[str, np.ndarray], user_idx: int) -> np.ndarray:
    indptr = cache["gt_indptr"]
    return cache["gt_items"][indptr[user_idx] : indptr[user_idx + 1]]


def empty_metric_sums() -> dict[str, float]:
    return {"NDCG@10": 0.0, "Recall@10": 0.0, "HR@10": 0.0, "MRR": 0.0}


def accumulate_metrics(
    sums: dict[str, float],
    ranked_items: np.ndarray,
    gt_items: np.ndarray,
) -> None:
    gt = set(int(item) for item in gt_items)
    if not gt:
        return
    top10 = [int(item) for item in ranked_items[:10]]
    hits = [1.0 if item in gt else 0.0 for item in top10]
    sums["HR@10"] += 1.0 if any(hits) else 0.0
    sums["Recall@10"] += sum(hits) / len(gt)
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt), 10)))
    sums["NDCG@10"] += dcg / idcg if idcg > 0 else 0.0
    mrr = 0.0
    for rank, item in enumerate(ranked_items):
        if int(item) in gt:
            mrr = 1.0 / (rank + 1)
            break
    sums["MRR"] += mrr


def feature_importance_rows(model, dataset: str, dataset_slug: str, seed: int) -> list[dict[str, Any]]:
    classifier = model.named_steps["sgdclassifier"]
    coefs = classifier.coef_[0]
    return [
        {
            "dataset": dataset,
            "dataset_slug": dataset_slug,
            "seed": seed,
            "feature": name,
            "coef": float(coef),
            "abs_coef": float(abs(coef)),
        }
        for name, coef in zip(FEATURE_NAMES, coefs)
    ]


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
                "mean_router_NDCG@10": float(np.mean([x["test_NDCG@10"] for x in items])),
                "std_router_NDCG@10": float(np.std([x["test_NDCG@10"] for x in items])),
                "mean_best_baseline_NDCG@10": float(np.mean([x["best_baseline_NDCG@10"] for x in items])),
                "mean_delta_vs_best_baseline": float(np.mean([x["delta_vs_best_baseline"] for x in items])),
                "mean_router_Recall@10": float(np.mean([x["test_Recall@10"] for x in items])),
                "mean_router_MRR": float(np.mean([x["test_MRR"] for x in items])),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V2 Router Results",
        "",
        "**Models:** soft blend grid over cached top-100 expert candidates",
        "",
        "| Dataset | Method | Seeds | Router NDCG@10 | Best Baseline NDCG@10 | Delta | Router Recall@10 | Router MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n_seeds']} | "
            f"{row['mean_router_NDCG@10']:.6f} | {row['mean_best_baseline_NDCG@10']:.6f} | "
            f"{row['mean_delta_vs_best_baseline']:.6f} | "
            f"{row['mean_router_Recall@10']:.6f} | {row['mean_router_MRR']:.6f} |"
        )
    lines.extend(["", "## Per-Seed", ""])
    lines.extend(
        [
            "| Dataset | Seed | Method | Weights | Router NDCG@10 | Best Baseline | Best Baseline NDCG@10 | Delta |",
            "|---|---:|---|---|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['method']} | `{row['weights']}` | "
            f"{row['test_NDCG@10']:.6f} | "
            f"{row['best_baseline']} | {row['best_baseline_NDCG@10']:.6f} | "
            f"{row['delta_vs_best_baseline']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_config(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _resolve_path(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent.parent / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


if __name__ == "__main__":
    main()
