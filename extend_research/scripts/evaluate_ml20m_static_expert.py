from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTEND_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SRC = REPO_ROOT / "src" / "benchmark"
sys.path.insert(0, str(EXTEND_ROOT / "src"))
sys.path.insert(0, str(BENCHMARK_SRC))

from evaluate import evaluate_model  # noqa: E402


DEFAULT_PROPAGATION_LAYERS = {
    "Amazon": {"M7": 3, "R1": 2, "R1-plus": 2},
    "ML-1M": {"M7": 3, "R1": 3, "R1-plus": 3},
    "ML-20M": {"M7": 3, "R1": 2, "R1-plus": 3},
    "ML20M-sub163": {"M7": 3, "R1": 3, "R1-plus": 3},
}


class FlexibleInteractionData:
    """InteractionData-compatible loader that accepts optional ID maps."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.train_df = pd.read_csv(data_dir / "train.csv")
        self.val_df = pd.read_csv(data_dir / "val.csv")
        self.test_df = pd.read_csv(data_dir / "test.csv")

        with (data_dir / "stats.json").open("r", encoding="utf-8") as handle:
            self.stats = json.load(handle)

        self.n_users = int(self.stats["n_users"])
        self.n_items = int(self.stats["n_items"])
        self.item_map = self._load_map_or_identity("item_map.json", self.n_items)
        self.user_map = self._load_map_or_identity("user_map.json", self.n_users)

        self.train_user_items = self._build_user_items(self.train_df)
        self.val_user_items = self._build_user_items(self.val_df)
        self.test_user_items = self._build_user_items(self.test_df)

    def _load_map_or_identity(self, name: str, size: int) -> dict[int, int]:
        path = self.data_dir / name
        if not path.exists():
            return {idx: idx for idx in range(size)}
        with path.open("r", encoding="utf-8") as handle:
            return {int(key): int(value) for key, value in json.load(handle).items()}

    @staticmethod
    def _build_user_items(df: pd.DataFrame) -> dict[int, set[int]]:
        user_items: dict[int, set[int]] = {}
        for uid, group in df.groupby("userId"):
            user_items[int(uid)] = set(int(item) for item in group["movieId"].tolist())
        return user_items

    def get_sparse_adj(self) -> sp.coo_matrix:
        users = self.train_df["userId"].to_numpy()
        items = self.train_df["movieId"].to_numpy() + self.n_users
        rows = np.concatenate([users, items])
        cols = np.concatenate([items, users])
        vals = np.ones(len(rows), dtype=np.float32)
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items),
        )

    def get_norm_adj(self) -> torch.Tensor:
        adj = self.get_sparse_adj()
        degree = np.array(adj.sum(axis=1)).reshape(-1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        norm_adj = (d_mat @ adj @ d_mat).tocoo()
        indices = torch.LongTensor(np.stack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).coalesce()


class StaticEmbeddingModel:
    """Minimal predict-only wrapper for precomputed user/item embeddings."""

    def __init__(self, user_embeds: torch.Tensor, item_embeds: torch.Tensor, device: str):
        self.user_embeds = user_embeds.to(device)
        self.item_embeds = item_embeds.to(device)
        self.device = device

    def eval(self) -> None:
        return None

    @torch.no_grad()
    def predict(self, user_ids: torch.Tensor) -> torch.Tensor:
        users = self.user_embeds[user_ids.to(self.device)]
        return users @ self.item_embeds.T


def lightgcn_propagate(
    user_embeds: torch.Tensor,
    item_embeds: torch.Tensor,
    norm_adj: torch.Tensor,
    n_layers: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match RLMRec's inference path: sum initial embeddings plus L graph hops."""
    user_count = user_embeds.shape[0]
    embeds = torch.cat([user_embeds, item_embeds], dim=0).to(device)
    norm_adj = norm_adj.to(device)
    final_embeds = embeds

    for _ in range(n_layers):
        embeds = torch.sparse.mm(norm_adj, embeds)
        final_embeds = final_embeds + embeds

    return final_embeds[:user_count].cpu(), final_embeds[user_count:].cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate available expert checkpoints with raw or LightGCN-propagated embeddings."
    )
    parser.add_argument("--checkpoint-config", type=Path, default=Path("configs/ml20m_checkpoint_paths.json"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/ml20m_data_paths.json"))
    parser.add_argument("--expert", choices=["M7", "R1", "R1-plus"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--embedding-source",
        choices=["user_item", "prf"],
        default="user_item",
        help="For R1, 'prf' uses prf_embeds split into users/items. M7/R1-plus only support user_item.",
    )
    parser.add_argument(
        "--scoring",
        choices=["lightgcn", "raw"],
        default="lightgcn",
        help="'lightgcn' reproduces RLMRec inference by propagating over the train graph; 'raw' is diagnostic only.",
    )
    parser.add_argument(
        "--propagation-layers",
        type=int,
        default=None,
        help="Override LightGCN propagation depth. Defaults are dataset/expert-specific.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_config_path = _resolve_config(args.checkpoint_config)
    data_config_path = _resolve_config(args.data_config)
    checkpoint_config = _read_json(checkpoint_config_path)
    data_config = _read_json(data_config_path)

    data_dir = _resolve_extend_path(data_config["data_dir"])
    checkpoint_path = _resolve_extend_path(
        checkpoint_config["experts"][args.expert]["paths"][str(args.seed)]
    )

    interaction_data = FlexibleInteractionData(data_dir=data_dir)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    user_embeds, item_embeds = _extract_embeddings(
        state=state,
        expert=args.expert,
        source=args.embedding_source,
        n_users=interaction_data.n_users,
        n_items=interaction_data.n_items,
        data_config=data_config,
    )

    propagation_layers = args.propagation_layers
    if propagation_layers is None:
        propagation_layers = _default_propagation_layers(data_config, args.expert)

    if args.scoring == "lightgcn":
        user_embeds, item_embeds = lightgcn_propagate(
            user_embeds=user_embeds,
            item_embeds=item_embeds,
            norm_adj=interaction_data.get_norm_adj(),
            n_layers=propagation_layers,
            device=args.device,
        )

    model = StaticEmbeddingModel(user_embeds=user_embeds, item_embeds=item_embeds, device=args.device)
    metrics = evaluate_model(
        model,
        interaction_data,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
    )

    result = {
        "dataset": data_config["dataset"],
        "expert": args.expert,
        "seed": args.seed,
        "split": args.split,
        "embedding_source": args.embedding_source,
        "scoring": args.scoring,
        "propagation_layers": propagation_layers if args.scoring == "lightgcn" else 0,
        "checkpoint": str(checkpoint_path),
        "checkpoint_config": str(checkpoint_config_path),
        "data_config": str(data_config_path),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }

    dataset_slug = _dataset_slug(data_config)
    output_path = args.output or (
        EXTEND_ROOT
        / "results"
        / "dts_v1b"
        / "static_eval"
        / dataset_slug
        / args.expert.lower().replace("-", "")
        / (
            f"seed-{args.seed}-{args.embedding_source}-{args.scoring}"
            f"-L{result['propagation_layers']}-{args.split}.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(
        f"{data_config['dataset']} {args.expert} seed={args.seed} source={args.embedding_source} "
        f"scoring={args.scoring} layers={result['propagation_layers']} split={args.split} "
        f"NDCG@10={metrics.get('NDCG@10', 0):.6f} "
        f"Recall@10={metrics.get('Recall@10', 0):.6f} "
        f"MRR={metrics.get('MRR', 0):.6f}"
    )
    print(f"Wrote {output_path}")


def _extract_embeddings(
    state: dict,
    expert: str,
    source: str,
    n_users: int,
    n_items: int,
    data_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expert == "M7":
        if source != "user_item":
            raise ValueError("M7 only supports --embedding-source user_item")
        required = [
            "user_emb.weight",
            "item_emb.weight",
            "feature_proj.0.weight",
            "feature_proj.0.bias",
            "feature_proj.2.weight",
            "feature_proj.2.bias",
        ]
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"M7 checkpoint missing keys: {missing}")
        user_embeds = state["user_emb.weight"].detach().float()
        base_item_embeds = state["item_emb.weight"].detach().float()
        item_features = torch.from_numpy(_load_profile_mood_features(data_config)).float()
        hidden = F.linear(item_features, state["feature_proj.0.weight"].float(), state["feature_proj.0.bias"].float())
        hidden = F.relu(hidden)
        projected = F.linear(hidden, state["feature_proj.2.weight"].float(), state["feature_proj.2.bias"].float())
        item_embeds = base_item_embeds + projected
        _validate_embedding_shapes(user_embeds, item_embeds, n_users, n_items)
        return user_embeds, item_embeds

    if source == "prf":
        if "prf_embeds" not in state:
            raise ValueError("Checkpoint does not contain prf_embeds")
        prf = state["prf_embeds"].detach().float()
        if prf.shape[0] != n_users + n_items:
            raise ValueError(f"Expected prf_embeds first dim {n_users + n_items}, got {prf.shape[0]}")
        return prf[:n_users], prf[n_users:]

    if "user_embeds" not in state or "item_embeds" not in state:
        raise ValueError("Checkpoint does not contain user_embeds and item_embeds")
    user_embeds = state["user_embeds"].detach().float()
    item_embeds = state["item_embeds"].detach().float()
    _validate_embedding_shapes(user_embeds, item_embeds, n_users, n_items)
    return user_embeds, item_embeds


def _validate_embedding_shapes(
    user_embeds: torch.Tensor,
    item_embeds: torch.Tensor,
    n_users: int,
    n_items: int,
) -> None:
    if user_embeds.shape[0] != n_users:
        raise ValueError(f"Expected {n_users} users, got {user_embeds.shape[0]}")
    if item_embeds.shape[0] != n_items:
        raise ValueError(f"Expected {n_items} items, got {item_embeds.shape[0]}")


def _load_profile_mood_features(data_config: dict) -> np.ndarray:
    data_dir = _resolve_extend_path(data_config["data_dir"])
    embedding_dir = _resolve_extend_path(data_config["embedding_dir"])

    with (data_dir / "stats.json").open("r", encoding="utf-8") as handle:
        n_items = int(json.load(handle)["n_items"])

    item_map_path = data_dir / "item_map.json"
    if item_map_path.exists():
        with item_map_path.open("r", encoding="utf-8") as handle:
            item_map = {int(key): int(value) for key, value in json.load(handle).items()}
    else:
        item_map = {idx: idx for idx in range(n_items)}

    with (embedding_dir / "movie_id_index.json").open("r", encoding="utf-8") as handle:
        movie_ids = json.load(handle)

    movie_id_to_row = {int(movie_id): idx for idx, movie_id in enumerate(movie_ids)}
    profile = np.load(embedding_dir / "profile_embeddings.npy")
    mood = np.load(embedding_dir / "mood_vectors.npy")
    raw = np.concatenate([profile, mood], axis=1).astype(np.float32, copy=False)
    aligned = np.zeros((n_items, raw.shape[1]), dtype=np.float32)

    for original_movie_id, contiguous_item_id in item_map.items():
        raw_idx = movie_id_to_row.get(original_movie_id)
        if raw_idx is not None:
            aligned[contiguous_item_id] = raw[raw_idx]

    return aligned


def _default_propagation_layers(data_config: dict, expert: str) -> int:
    dataset = data_config["dataset"]
    try:
        return DEFAULT_PROPAGATION_LAYERS[dataset][expert]
    except KeyError as exc:
        raise ValueError(f"No default propagation depth for dataset={dataset}, expert={expert}") from exc


def _dataset_slug(data_config: dict) -> str:
    raw = data_config.get("hf_dataset_name") or data_config["dataset"]
    return str(raw).lower().replace(" ", "_").replace("-", "_")


def _resolve_config(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (EXTEND_ROOT / path).resolve()


def _resolve_extend_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (EXTEND_ROOT / candidate).resolve()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
