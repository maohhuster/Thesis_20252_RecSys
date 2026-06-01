from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    density: float
    result_dir: str
    file_suffix: str


@dataclass(frozen=True)
class ExpertResult:
    dataset: str
    expert: str
    density: float
    ndcg10: float
    recall10: float | None = None
    mrr: float | None = None
    source_file: str | None = None


EXPERT_TO_FILE_STEM = {
    "R1": "r1",
    "R1-plus": "r1plus",
    "R2": "r2",
    "R3": "r3",
}


def load_dataset_specs(config: dict[str, Any]) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name=item["name"],
            density=float(item["density"]),
            result_dir=item["result_dir"],
            file_suffix=item.get("file_suffix", ""),
        )
        for item in config["datasets"]
    ]


def load_tier3_results(config: dict[str, Any], config_path: Path) -> list[ExpertResult]:
    benchmark_root = (config_path.parent / config["benchmark_root"]).resolve()
    rows: list[ExpertResult] = []

    for dataset in load_dataset_specs(config):
        m7_ndcg: float | None = None
        dataset_rows: list[ExpertResult] = []

        for expert, stem in EXPERT_TO_FILE_STEM.items():
            path = benchmark_root / dataset.result_dir / f"{stem}{dataset.file_suffix}_metrics.json"
            if not path.exists():
                continue

            data = _read_json(path)
            summary = data.get("summary", {})
            ndcg = _metric_mean(summary, "NDCG@10")
            recall = _metric_mean(summary, "Recall@10")
            mrr = _metric_mean(summary, "MRR")

            if m7_ndcg is None:
                m7_ndcg = data.get("vs_baseline", {}).get("baseline_NDCG10")

            dataset_rows.append(
                ExpertResult(
                    dataset=dataset.name,
                    expert=expert,
                    density=dataset.density,
                    ndcg10=ndcg,
                    recall10=recall,
                    mrr=mrr,
                    source_file=str(path),
                )
            )

        if m7_ndcg is None:
            raise ValueError(f"Could not infer M7 baseline NDCG for {dataset.name}")

        rows.append(
            ExpertResult(
                dataset=dataset.name,
                expert="M7",
                density=dataset.density,
                ndcg10=float(m7_ndcg),
                source_file="vs_baseline.baseline_NDCG10",
            )
        )
        rows.extend(dataset_rows)

    return rows


def load_ml20m_cold_start(config: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    benchmark_root = (config_path.parent / config["benchmark_root"]).resolve()
    path = benchmark_root / "results" / "cold_start_5seeds.json"
    if not path.exists():
        return None
    return _read_json(path)


def _metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
