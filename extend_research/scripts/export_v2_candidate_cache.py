from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXTEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTEND_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "benchmark"))

from evaluate_ml20m_static_expert import (  # noqa: E402
    FlexibleInteractionData,
    _read_json,
    _resolve_config,
    _resolve_extend_path,
)
from run_dts_v1b_routing import (  # noqa: E402
    _item_degree,
    load_expert_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V2 top-K candidate/score cache.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_candidate_cache.json"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset slugs to export.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seeds to export.")
    parser.add_argument("--splits", nargs="*", default=None, choices=["val", "test"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_config(args.config)
    config = _read_json(config_path)
    output_dir = _resolve_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = args.top_k or int(config["top_k"])
    batch_size = args.batch_size or int(config["batch_size"])
    device = args.device or config.get("device", "cpu")
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    splits = args.splits or list(config["splits"])
    selected_datasets = set(args.datasets) if args.datasets else None

    manifest_rows: list[dict[str, Any]] = []

    for dataset_spec in config["datasets"]:
        if selected_datasets and dataset_spec["slug"] not in selected_datasets:
            continue
        if not dataset_spec.get("enabled", True):
            print(f"Skipping {dataset_spec['slug']}: {dataset_spec.get('skip_reason', 'disabled')}")
            continue

        print(f"=== {dataset_spec['name']} ===", flush=True)
        checkpoint_config = _read_json(_resolve_config(Path(dataset_spec["checkpoint_config"])))
        data_config = _read_json(_resolve_config(Path(dataset_spec["data_config"])))
        interaction_data = FlexibleInteractionData(_resolve_extend_path(data_config["data_dir"]))
        item_degree = _item_degree(interaction_data).astype(np.int32)
        user_history_len = _user_history_lengths(interaction_data).astype(np.int32)

        for seed in seeds:
            print(f"  seed={seed}: loading propagated expert embeddings", flush=True)
            expert_embeddings = load_expert_embeddings(
                checkpoint_config=checkpoint_config,
                data_config=data_config,
                interaction_data=interaction_data,
                seed=seed,
                device=device,
            )

            for split in splits:
                out_path = (
                    output_dir
                    / dataset_spec["slug"]
                    / f"seed-{seed}"
                    / f"{split}_top{top_k}.npz"
                )
                if out_path.exists() and not args.force:
                    print(f"  seed={seed} split={split}: exists, skipping {out_path}", flush=True)
                    manifest_rows.append(_manifest_row(dataset_spec, seed, split, top_k, out_path, skipped=True))
                    continue

                print(f"  seed={seed} split={split}: exporting top-{top_k}", flush=True)
                export_split_cache(
                    output_path=out_path,
                    dataset_spec=dataset_spec,
                    seed=seed,
                    split=split,
                    top_k=top_k,
                    batch_size=batch_size,
                    device=device,
                    expert_embeddings=expert_embeddings,
                    interaction_data=interaction_data,
                    item_degree=item_degree,
                    user_history_len=user_history_len,
                )
                manifest_rows.append(_manifest_row(dataset_spec, seed, split, top_k, out_path, skipped=False))

    _write_manifest(output_dir / "manifest.json", config, manifest_rows)
    print(f"Wrote V2 candidate cache manifest to {output_dir / 'manifest.json'}", flush=True)


@torch.no_grad()
def export_split_cache(
    output_path: Path,
    dataset_spec: dict[str, Any],
    seed: int,
    split: str,
    top_k: int,
    batch_size: int,
    device: str,
    expert_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    interaction_data: FlexibleInteractionData,
    item_degree: np.ndarray,
    user_history_len: np.ndarray,
) -> None:
    eval_user_items = interaction_data.val_user_items if split == "val" else interaction_data.test_user_items
    users = np.array(sorted(eval_user_items), dtype=np.int64)
    n_users = len(users)
    train_user_items = interaction_data.train_user_items

    arrays: dict[str, np.ndarray] = {
        "users": users,
        "user_history_len": user_history_len[users],
        "item_degree": item_degree,
    }
    gt_indptr, gt_items = _ground_truth_csr(users.tolist(), eval_user_items)
    arrays["gt_indptr"] = gt_indptr
    arrays["gt_items"] = gt_items

    for expert in expert_embeddings:
        arrays[f"{expert_key(expert)}_top_items"] = np.empty((n_users, top_k), dtype=np.int32)
        arrays[f"{expert_key(expert)}_top_scores"] = np.empty((n_users, top_k), dtype=np.float32)

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_users = users[start:end].tolist()
        user_tensor = torch.LongTensor(batch_users).to(device)

        for expert, (user_embeds, item_embeds) in expert_embeddings.items():
            raw = user_embeds[user_tensor] @ item_embeds.T
            mean = raw.mean(dim=1, keepdim=True)
            std = raw.std(dim=1, keepdim=True).clamp_min(1e-6)
            scores = (raw - mean) / std

            for row_idx, user_id in enumerate(batch_users):
                train_items = train_user_items.get(int(user_id), set())
                if train_items:
                    scores[row_idx, torch.LongTensor(list(train_items)).to(device)] = -1e8

            top_scores, top_items = torch.topk(scores, k=top_k, dim=1)
            key = expert_key(expert)
            arrays[f"{key}_top_items"][start:end] = top_items.cpu().numpy().astype(np.int32)
            arrays[f"{key}_top_scores"][start:end] = top_scores.cpu().numpy().astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset_spec["name"],
        "dataset_slug": dataset_spec["slug"],
        "seed": seed,
        "split": split,
        "top_k": top_k,
        "n_eval_users": n_users,
        "n_items": int(interaction_data.n_items),
        "experts": list(expert_embeddings),
        "score_normalization": "per-user z-score before train-history masking",
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(output_path, **arrays)


def expert_key(expert: str) -> str:
    if expert == "R1-plus":
        return "r1plus"
    return expert.lower().replace("-", "_")


def _user_history_lengths(interaction_data: FlexibleInteractionData) -> np.ndarray:
    lengths = np.zeros(interaction_data.n_users, dtype=np.int32)
    for user_id, items in interaction_data.train_user_items.items():
        lengths[int(user_id)] = len(items)
    return lengths


def _ground_truth_csr(
    users: list[int],
    eval_user_items: dict[int, set[int]],
) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(users) + 1, dtype=np.int64)
    flat: list[int] = []
    for idx, user_id in enumerate(users):
        items = sorted(eval_user_items.get(user_id, set()))
        flat.extend(items)
        indptr[idx + 1] = len(flat)
    return indptr, np.array(flat, dtype=np.int32)


def _manifest_row(
    dataset_spec: dict[str, Any],
    seed: int,
    split: str,
    top_k: int,
    path: Path,
    skipped: bool,
) -> dict[str, Any]:
    return {
        "dataset": dataset_spec["name"],
        "dataset_slug": dataset_spec["slug"],
        "seed": seed,
        "split": split,
        "top_k": top_k,
        "path": str(path),
        "skipped_existing": skipped,
    }


def _write_manifest(
    path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "experiment_name": config["experiment_name"],
        "description": config["description"],
        "cache_files": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_dir = Path(config["output_dir"])
    if output_dir.is_absolute():
        return output_dir
    return (config_path.parent.parent / output_dir).resolve()


if __name__ == "__main__":
    main()
