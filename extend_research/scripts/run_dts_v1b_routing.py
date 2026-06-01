from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXTEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTEND_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "benchmark"))

from evaluate_ml20m_static_expert import (  # noqa: E402
    FlexibleInteractionData,
    _default_propagation_layers,
    _extract_embeddings,
    _read_json,
    _resolve_config,
    _resolve_extend_path,
    lightgcn_propagate,
)


@dataclass(frozen=True)
class RoutingPolicy:
    name: str
    kind: str
    experts: tuple[str, ...]
    thresholds: tuple[int, ...] = ()

    def route_items(self, item_degree: np.ndarray) -> np.ndarray:
        if self.kind == "fixed":
            return np.full(item_degree.shape, self.experts[0], dtype=object)
        if self.kind == "hard":
            sparse, dense = self.experts
            threshold = self.thresholds[0]
            return np.where(item_degree < threshold, sparse, dense).astype(object)
        if self.kind == "two_threshold":
            low, mid, high = self.experts
            low_t, high_t = self.thresholds
            route = np.full(item_degree.shape, high, dtype=object)
            route[item_degree < high_t] = mid
            route[item_degree < low_t] = low
            return route
        raise ValueError(f"Unknown policy kind: {self.kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DTS-v1b per-item threshold reranking.")
    parser.add_argument("--config", type=Path, default=Path("configs/dts_v1b.json"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs to run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seeds to run.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config(args.config)
    config = _read_json(config_path)
    output_dir = _resolve_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or config.get("device", "cpu")
    batch_size = args.batch_size or int(config.get("batch_size", 512))
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    selected_datasets = set(args.datasets) if args.datasets else None
    policies = build_policies(config)

    all_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for dataset_spec in config["datasets"]:
        if selected_datasets and dataset_spec["slug"] not in selected_datasets:
            continue
        if not dataset_spec.get("enabled", True):
            print(f"Skipping {dataset_spec['slug']}: {dataset_spec.get('skip_reason', 'disabled')}")
            continue

        print(f"=== {dataset_spec['name']} ===")
        checkpoint_config = _read_json(_resolve_config(Path(dataset_spec["checkpoint_config"])))
        data_config = _read_json(_resolve_config(Path(dataset_spec["data_config"])))
        interaction_data = FlexibleInteractionData(_resolve_extend_path(data_config["data_dir"]))
        item_degree = _item_degree(interaction_data)

        for seed in seeds:
            print(f"  seed={seed}: loading experts")
            expert_embeddings = load_expert_embeddings(
                checkpoint_config=checkpoint_config,
                data_config=data_config,
                interaction_data=interaction_data,
                seed=seed,
                device=device,
            )

            val_rows = evaluate_policies(
                dataset=dataset_spec["name"],
                dataset_slug=dataset_spec["slug"],
                seed=seed,
                split="val",
                policies=policies,
                expert_embeddings=expert_embeddings,
                interaction_data=interaction_data,
                item_degree=item_degree,
                top_k=[int(k) for k in config["top_k"]],
                batch_size=batch_size,
                device=device,
            )
            best_val = max(val_rows, key=lambda row: (row["NDCG@10"], row["Recall@10"], row["MRR"]))

            test_rows = evaluate_policies(
                dataset=dataset_spec["name"],
                dataset_slug=dataset_spec["slug"],
                seed=seed,
                split="test",
                policies=policies,
                expert_embeddings=expert_embeddings,
                interaction_data=interaction_data,
                item_degree=item_degree,
                top_k=[int(k) for k in config["top_k"]],
                batch_size=batch_size,
                device=device,
            )
            test_by_name = {row["policy"]: row for row in test_rows}
            selected_test = test_by_name[best_val["policy"]]
            selection_rows.append(
                {
                    "dataset": dataset_spec["name"],
                    "dataset_slug": dataset_spec["slug"],
                    "seed": seed,
                    "selected_policy": best_val["policy"],
                    "val_NDCG@10": best_val["NDCG@10"],
                    "test_NDCG@10": selected_test["NDCG@10"],
                    "test_Recall@10": selected_test["Recall@10"],
                    "test_MRR": selected_test["MRR"],
                }
            )
            all_rows.extend(val_rows)
            all_rows.extend(test_rows)
            print(
                f"  seed={seed}: selected {best_val['policy']} "
                f"val NDCG@10={best_val['NDCG@10']:.6f}, "
                f"test NDCG@10={selected_test['NDCG@10']:.6f}"
            )

    _write_csv(output_dir / "policy_metrics.csv", all_rows)
    _write_csv(output_dir / "selected_policy_by_seed.csv", selection_rows)
    summary_rows = summarize_selection(selection_rows)
    _write_csv(output_dir / "selected_policy_summary.csv", summary_rows)
    _write_markdown(output_dir / "DTS_V1b_Routing_Results.md", summary_rows, selection_rows)
    print(f"Wrote DTS-v1b routing outputs to {output_dir}")


def build_policies(config: dict[str, Any]) -> list[RoutingPolicy]:
    policies = [
        RoutingPolicy(name=f"{expert}-only", kind="fixed", experts=(expert,))
        for expert in config["experts"]
    ]
    for threshold in config["thresholds"]:
        for sparse_expert in ["R1", "R1-plus"]:
            policies.append(
                RoutingPolicy(
                    name=f"DTS-hard-{sparse_expert}-T{threshold}",
                    kind="hard",
                    experts=(sparse_expert, "M7"),
                    thresholds=(int(threshold),),
                )
            )
    for low_t, high_t in config["two_thresholds"]:
        policies.append(
            RoutingPolicy(
                name=f"DTS-2T-R1-R1plus-M7-T{low_t}-T{high_t}",
                kind="two_threshold",
                experts=("R1", "R1-plus", "M7"),
                thresholds=(int(low_t), int(high_t)),
            )
        )
    return policies


def load_expert_embeddings(
    checkpoint_config: dict[str, Any],
    data_config: dict[str, Any],
    interaction_data: FlexibleInteractionData,
    seed: int,
    device: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    norm_adj = interaction_data.get_norm_adj()
    for expert in checkpoint_config["experts"]:
        checkpoint_path = _resolve_extend_path(checkpoint_config["experts"][expert]["paths"][str(seed)])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        user_embeds, item_embeds = _extract_embeddings(
            state=state,
            expert=expert,
            source="user_item",
            n_users=interaction_data.n_users,
            n_items=interaction_data.n_items,
            data_config=data_config,
        )
        layers = _default_propagation_layers(data_config, expert)
        user_embeds, item_embeds = lightgcn_propagate(
            user_embeds=user_embeds,
            item_embeds=item_embeds,
            norm_adj=norm_adj,
            n_layers=layers,
            device=device,
        )
        embeddings[expert] = (user_embeds.to(device), item_embeds.to(device))
    return embeddings


@torch.no_grad()
def evaluate_policies(
    dataset: str,
    dataset_slug: str,
    seed: int,
    split: str,
    policies: list[RoutingPolicy],
    expert_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    interaction_data: FlexibleInteractionData,
    item_degree: np.ndarray,
    top_k: list[int],
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    eval_user_items = interaction_data.val_user_items if split == "val" else interaction_data.test_user_items
    train_user_items = interaction_data.train_user_items
    eval_users = sorted(eval_user_items)
    max_k = max(top_k)

    policy_routes = {
        policy.name: {
            expert: torch.from_numpy(np.where(policy.route_items(item_degree) == expert)[0]).long().to(device)
            for expert in expert_embeddings
        }
        for policy in policies
    }
    metric_sums = {policy.name: _empty_metric_sums(top_k) for policy in policies}

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start : start + batch_size]
        user_tensor = torch.LongTensor(batch_users).to(device)
        expert_scores = _score_experts(expert_embeddings, user_tensor)

        for row_idx, user_id in enumerate(batch_users):
            train_items = train_user_items.get(user_id, set())
            if train_items:
                idx = torch.LongTensor(list(train_items)).to(device)
                for scores in expert_scores.values():
                    scores[row_idx, idx] = -1e8

        for policy in policies:
            routed_scores = torch.empty_like(next(iter(expert_scores.values())))
            for expert, idx in policy_routes[policy.name].items():
                if idx.numel() > 0:
                    routed_scores[:, idx] = expert_scores[expert][:, idx]
            _, top_items = torch.topk(routed_scores, k=max_k, dim=1)
            top_items_np = top_items.cpu().numpy()
            _accumulate_metrics(
                metric_sums[policy.name],
                top_items_np,
                batch_users,
                eval_user_items,
                top_k,
            )

    rows = []
    n_eval_users = len(eval_users)
    for policy in policies:
        sums = metric_sums[policy.name]
        row = {
            "dataset": dataset,
            "dataset_slug": dataset_slug,
            "seed": seed,
            "split": split,
            "policy": policy.name,
            "n_eval_users": n_eval_users,
        }
        for metric, value in sums.items():
            row[metric] = value / n_eval_users if n_eval_users else 0.0
        rows.append(row)
    return rows


def _score_experts(
    expert_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    user_tensor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scores = {}
    for expert, (user_embeds, item_embeds) in expert_embeddings.items():
        raw = user_embeds[user_tensor] @ item_embeds.T
        mean = raw.mean(dim=1, keepdim=True)
        std = raw.std(dim=1, keepdim=True).clamp_min(1e-6)
        scores[expert] = (raw - mean) / std
    return scores


def _empty_metric_sums(top_k: list[int]) -> dict[str, float]:
    sums = {"MRR": 0.0}
    for k in top_k:
        sums[f"NDCG@{k}"] = 0.0
        sums[f"Recall@{k}"] = 0.0
        sums[f"HR@{k}"] = 0.0
    return sums


def _accumulate_metrics(
    sums: dict[str, float],
    top_items: np.ndarray,
    users: list[int],
    eval_user_items: dict[int, set[int]],
    top_k: list[int],
) -> None:
    for row_idx, user_id in enumerate(users):
        gt = eval_user_items[user_id]
        if not gt:
            continue
        ranked = top_items[row_idx]
        for k in top_k:
            top = ranked[:k]
            hits = [1.0 if int(item) in gt else 0.0 for item in top]
            sums[f"HR@{k}"] += 1.0 if any(hits) else 0.0
            sums[f"Recall@{k}"] += sum(hits) / len(gt)
            dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gt), k)))
            sums[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0
        mrr = 0.0
        for rank, item in enumerate(ranked):
            if int(item) in gt:
                mrr = 1.0 / (rank + 1)
                break
        sums["MRR"] += mrr


def _item_degree(interaction_data: FlexibleInteractionData) -> np.ndarray:
    degree = np.zeros(interaction_data.n_items, dtype=np.int64)
    counts = interaction_data.train_df["movieId"].value_counts()
    for item, count in counts.items():
        degree[int(item)] = int(count)
    return degree


def summarize_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["dataset_slug"]), []).append(row)

    summary = []
    for (dataset, dataset_slug), items in sorted(grouped.items()):
        selected_counts: dict[str, int] = {}
        for item in items:
            selected_counts[item["selected_policy"]] = selected_counts.get(item["selected_policy"], 0) + 1
        summary.append(
            {
                "dataset": dataset,
                "dataset_slug": dataset_slug,
                "n_seeds": len(items),
                "mean_val_NDCG@10": float(np.mean([item["val_NDCG@10"] for item in items])),
                "std_val_NDCG@10": float(np.std([item["val_NDCG@10"] for item in items])),
                "mean_test_NDCG@10": float(np.mean([item["test_NDCG@10"] for item in items])),
                "std_test_NDCG@10": float(np.std([item["test_NDCG@10"] for item in items])),
                "mean_test_Recall@10": float(np.mean([item["test_Recall@10"] for item in items])),
                "mean_test_MRR": float(np.mean([item["test_MRR"] for item in items])),
                "selected_policy_counts": json.dumps(selected_counts, sort_keys=True),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# DTS-v1b Routing Results",
        "",
        "**Stage:** validation-selected per-item density threshold reranking",
        "",
        "| Dataset | Seeds | Mean Test NDCG@10 | Std | Mean Test Recall@10 | Mean Test MRR | Selected Policies |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_seeds']} | "
            f"{row['mean_test_NDCG@10']:.6f} | {row['std_test_NDCG@10']:.6f} | "
            f"{row['mean_test_Recall@10']:.6f} | {row['mean_test_MRR']:.6f} | "
            f"`{row['selected_policy_counts']}` |"
        )
    lines.extend(["", "## Per-Seed Selection", ""])
    lines.extend(
        [
            "| Dataset | Seed | Selected Policy | Val NDCG@10 | Test NDCG@10 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in selection_rows:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['selected_policy']} | "
            f"{row['val_NDCG@10']:.6f} | {row['test_NDCG@10']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_dir = Path(config["output_dir"])
    if output_dir.is_absolute():
        return output_dir
    return (config_path.parent.parent / output_dir).resolve()


if __name__ == "__main__":
    main()
