from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_v2_2_segments import METRIC_KEYS, ground_truth_items, rank_blend, read_selected_weights, user_metrics
from train_v2_router import EXPERT_KEYS, load_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap V2.2 paired user-level gains.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_3_bootstrap_significance.json"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
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
    n_bootstrap = args.n_bootstrap or int(config["n_bootstrap"])
    rng = np.random.default_rng(int(config["random_seed"]))

    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        if selected_datasets and dataset["slug"] not in selected_datasets:
            continue
        if not dataset.get("enabled", True):
            print(f"Skipping {dataset['slug']}: {dataset.get('skip_reason', 'disabled')}", flush=True)
            continue

        weights = selected_weights[dataset["slug"]]
        print(f"=== {dataset['name']} weights={weights} ===", flush=True)
        dataset_deltas: dict[str, list[np.ndarray]] = {key: [] for key in METRIC_KEYS}

        for seed in seeds:
            path = cache_dir / dataset["slug"] / f"seed-{seed}" / f"test_top{config['top_k']}.npz"
            if not path.exists():
                print(f"  seed={seed}: missing cache, skipping", flush=True)
                continue
            cache = load_cache(path)
            best_expert, per_user_deltas, per_user_blend, per_user_best = paired_user_deltas(
                cache=cache,
                weights=weights,
                missing_score=float(config["model"]["missing_score"]),
            )
            for metric in METRIC_KEYS:
                dataset_deltas[metric].append(per_user_deltas[metric])
            seed_rows.append(
                seed_result_row(
                    dataset=dataset,
                    seed=seed,
                    best_expert=best_expert,
                    per_user_deltas=per_user_deltas,
                    per_user_blend=per_user_blend,
                    per_user_best=per_user_best,
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                )
            )
            print(
                f"  seed={seed}: best={best_expert} "
                f"delta_NDCG@10={float(np.mean(per_user_deltas['NDCG@10'])):.6f}",
                flush=True,
            )

        for metric in METRIC_KEYS:
            if dataset_deltas[metric]:
                summary_rows.append(
                    summary_result_row(
                        dataset=dataset,
                        metric=metric,
                        deltas=np.concatenate(dataset_deltas[metric]),
                        n_seeds=len(dataset_deltas[metric]),
                        n_bootstrap=n_bootstrap,
                        rng=rng,
                    )
                )

    write_csv(output_dir / "bootstrap_by_seed.csv", seed_rows)
    write_csv(output_dir / "bootstrap_summary.csv", summary_rows)
    write_markdown(output_dir / "V2_3_Bootstrap_Significance.md", summary_rows, seed_rows)
    print(f"Wrote bootstrap outputs to {output_dir}", flush=True)


def paired_user_deltas(
    cache: dict[str, np.ndarray],
    weights: tuple[float, float, float],
    missing_score: float,
) -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    expert_user_metrics = {expert: {metric: [] for metric in METRIC_KEYS} for expert in EXPERT_KEYS}
    blend_user_metrics = {metric: [] for metric in METRIC_KEYS}

    for user_idx in range(len(cache["users"])):
        gt_items = ground_truth_items(cache, user_idx)
        blend_metrics = user_metrics(rank_blend(cache, user_idx, weights, missing_score), gt_items)
        for metric in METRIC_KEYS:
            blend_user_metrics[metric].append(blend_metrics[metric])

        for expert in EXPERT_KEYS:
            expert_metrics = user_metrics(cache[f"{expert}_top_items"][user_idx], gt_items)
            for metric in METRIC_KEYS:
                expert_user_metrics[expert][metric].append(expert_metrics[metric])

    blend_arrays = {metric: np.array(values, dtype=np.float64) for metric, values in blend_user_metrics.items()}
    expert_arrays = {
        expert: {metric: np.array(values, dtype=np.float64) for metric, values in metrics.items()}
        for expert, metrics in expert_user_metrics.items()
    }
    best_expert = max(
        EXPERT_KEYS,
        key=lambda expert: (
            float(np.mean(expert_arrays[expert]["NDCG@10"])),
            float(np.mean(expert_arrays[expert]["Recall@10"])),
            float(np.mean(expert_arrays[expert]["MRR"])),
        ),
    )
    best_arrays = expert_arrays[best_expert]
    deltas = {metric: blend_arrays[metric] - best_arrays[metric] for metric in METRIC_KEYS}
    return best_expert, deltas, blend_arrays, best_arrays


def seed_result_row(
    dataset: dict[str, Any],
    seed: int,
    best_expert: str,
    per_user_deltas: dict[str, np.ndarray],
    per_user_blend: dict[str, np.ndarray],
    per_user_best: dict[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ndcg_ci = bootstrap_ci(per_user_deltas["NDCG@10"], n_bootstrap=n_bootstrap, rng=rng)
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "seed": seed,
        "n_users": len(per_user_deltas["NDCG@10"]),
        "best_expert": best_expert,
        "blend_NDCG@10": float(np.mean(per_user_blend["NDCG@10"])),
        "best_expert_NDCG@10": float(np.mean(per_user_best["NDCG@10"])),
        "delta_NDCG@10": float(np.mean(per_user_deltas["NDCG@10"])),
        "delta_NDCG@10_ci_low": ndcg_ci[0],
        "delta_NDCG@10_ci_high": ndcg_ci[1],
        "prob_delta_NDCG@10_gt_0": ndcg_ci[2],
        "blend_Recall@10": float(np.mean(per_user_blend["Recall@10"])),
        "best_expert_Recall@10": float(np.mean(per_user_best["Recall@10"])),
        "delta_Recall@10": float(np.mean(per_user_deltas["Recall@10"])),
        "blend_MRR": float(np.mean(per_user_blend["MRR"])),
        "best_expert_MRR": float(np.mean(per_user_best["MRR"])),
        "delta_MRR": float(np.mean(per_user_deltas["MRR"])),
    }


def summary_result_row(
    dataset: dict[str, Any],
    metric: str,
    deltas: np.ndarray,
    n_seeds: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    low, high, prob_positive = bootstrap_ci(deltas, n_bootstrap=n_bootstrap, rng=rng)
    return {
        "dataset": dataset["name"],
        "dataset_slug": dataset["slug"],
        "metric": metric,
        "n_seeds": n_seeds,
        "n_users_total": len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": low,
        "ci95_high": high,
        "prob_delta_gt_0": prob_positive,
        "fraction_users_positive": float(np.mean(deltas > 0.0)),
        "fraction_users_negative": float(np.mean(deltas < 0.0)),
        "fraction_users_tied": float(np.mean(deltas == 0.0)),
    }


def bootstrap_ci(
    deltas: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(deltas)
    sample_means = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample_means[idx] = float(np.mean(deltas[sample_idx]))
    return (
        float(np.quantile(sample_means, 0.025)),
        float(np.quantile(sample_means, 0.975)),
        float(np.mean(sample_means > 0.0)),
    )


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
        "# V2.3 Bootstrap Significance",
        "",
        "## Summary",
        "",
        "| Dataset | Metric | Users | Mean Delta | 95% CI | P(delta > 0) | Users + / - / = |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['n_users_total']} | "
            f"{row['mean_delta']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['prob_delta_gt_0']:.4f} | "
            f"{row['fraction_users_positive']:.3f} / "
            f"{row['fraction_users_negative']:.3f} / "
            f"{row['fraction_users_tied']:.3f} |"
        )

    lines.extend(["", "## Per-Seed NDCG@10", ""])
    lines.extend(
        [
            "| Dataset | Seed | Best Expert | Blend | Best Expert | Delta | 95% CI | P(delta > 0) |",
            "|---|---:|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['best_expert']} | "
            f"{row['blend_NDCG@10']:.6f} | {row['best_expert_NDCG@10']:.6f} | "
            f"{row['delta_NDCG@10']:.6f} | "
            f"[{row['delta_NDCG@10_ci_low']:.6f}, {row['delta_NDCG@10_ci_high']:.6f}] | "
            f"{row['prob_delta_NDCG@10_gt_0']:.4f} |"
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
