from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extend_research.analysis.dts_v1_data import (  # noqa: E402
    ExpertResult,
    load_ml20m_cold_start,
    load_tier3_results,
)
from extend_research.evaluation.router_metrics import (  # noqa: E402
    PolicySummary,
    evaluate_fixed_experts,
    evaluate_threshold_policies,
    index_results,
)
from extend_research.selectors.threshold_selector import (  # noqa: E402
    build_threshold_policies,
    build_two_threshold_policies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DTS-v1 aggregate diagnostic.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dts_v1.json"),
        help="Path to the DTS-v1 JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)

    output_dir = _resolve_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_tier3_results(config, config_path)
    fixed_experts = ["M7", "R1", "R1-plus"]
    policies = build_threshold_policies(
        thresholds=[float(value) for value in config["thresholds"]],
        sparse_experts=list(config["primary_regularizers"]),
        dense_expert="M7",
    )
    policies.extend(build_two_threshold_policies(config.get("two_threshold_policies", [])))

    fixed_summaries = evaluate_fixed_experts(rows, fixed_experts)
    threshold_summaries = evaluate_threshold_policies(rows, policies, fixed_experts)
    all_summaries = fixed_summaries + threshold_summaries
    best_summary = max(all_summaries, key=lambda item: (item.mean_ndcg10, -item.mean_regret_vs_oracle))

    cold_start = load_ml20m_cold_start(config, config_path)
    bucket_rows = _extract_ml20m_bucket_rows(cold_start) if cold_start else []

    _write_expert_table(output_dir / "expert_table.csv", rows)
    _write_summary_table(output_dir / "policy_summary.csv", all_summaries)
    _write_decision_table(output_dir / "policy_decisions.csv", all_summaries)
    _write_bucket_table(output_dir / "ml20m_bucket_table.csv", bucket_rows)

    report = {
        "experiment_name": config["experiment_name"],
        "stage": "V1a aggregate diagnostic",
        "selection_metric": config["selection_metric"],
        "best_policy": _summary_to_dict(best_summary),
        "experts": [asdict(row) for row in rows],
        "policy_summaries": [_summary_to_dict(summary) for summary in all_summaries],
        "ml20m_bucket_rows": bucket_rows,
        "limitations": [
            "This is an aggregate diagnostic, not a real per-user reranking selector.",
            "M7 cross-density values are inferred from Tier-3 result JSON baseline_NDCG10 fields.",
            "Cold-start bucket diagnostics are available only for M1/M4/M7 in the released artifact.",
        ],
    }
    _write_json(output_dir / "dts_v1_report.json", report)
    _write_markdown_summary(output_dir / "DTS_V1_Results.md", report)

    print(f"Wrote DTS-v1 diagnostic outputs to {output_dir}")
    print(
        "Best aggregate policy: "
        f"{best_summary.method} "
        f"(mean NDCG@10={best_summary.mean_ndcg10:.6f}, "
        f"mean regret={best_summary.mean_regret_vs_oracle:.6f})"
    )


def _resolve_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_dir = Path(config["output_dir"])
    if output_dir.is_absolute():
        return output_dir
    return (config_path.parent.parent / output_dir).resolve()


def _write_expert_table(path: Path, rows: list[ExpertResult]) -> None:
    fieldnames = ["dataset", "density", "expert", "ndcg10", "recall10", "mrr", "source_file"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.dataset, item.expert)):
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "density": row.density,
                    "expert": row.expert,
                    "ndcg10": row.ndcg10,
                    "recall10": row.recall10,
                    "mrr": row.mrr,
                    "source_file": row.source_file,
                }
            )


def _write_summary_table(path: Path, summaries: list[PolicySummary]) -> None:
    fieldnames = ["method", "mean_ndcg10", "mean_regret_vs_oracle", "win_rate_vs_best_fixed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: (-item.mean_ndcg10, item.mean_regret_vs_oracle)):
            writer.writerow(
                {
                    "method": summary.method,
                    "mean_ndcg10": summary.mean_ndcg10,
                    "mean_regret_vs_oracle": summary.mean_regret_vs_oracle,
                    "win_rate_vs_best_fixed": summary.win_rate_vs_best_fixed,
                }
            )


def _write_decision_table(path: Path, summaries: list[PolicySummary]) -> None:
    fieldnames = [
        "method",
        "dataset",
        "density",
        "chosen_expert",
        "chosen_ndcg10",
        "oracle_expert",
        "oracle_ndcg10",
        "regret_vs_oracle",
        "best_fixed_expert",
        "best_fixed_ndcg10",
        "delta_vs_best_fixed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for decision in summary.decisions:
                writer.writerow(asdict(decision))


def _extract_ml20m_bucket_rows(cold_start: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_name in ["M1", "M4", "M7"]:
        config_data = cold_start.get("configs", {}).get(config_name)
        if not config_data:
            continue
        for bucket, metrics in config_data.get("buckets", {}).items():
            row: dict[str, Any] = {
                "config": config_name,
                "bucket": bucket,
                "n_users": metrics.get("n_users"),
                "n_gt_items": metrics.get("n_gt_items"),
            }
            for metric in ["NDCG@100", "Recall@100", "NDCG@1000", "Recall@1000", "MRR_full"]:
                metric_value = metrics.get(metric)
                row[metric] = metric_value.get("mean") if isinstance(metric_value, dict) else None
            rows.append(row)
    return rows


def _write_bucket_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config",
        "bucket",
        "n_users",
        "n_gt_items",
        "NDCG@100",
        "Recall@100",
        "NDCG@1000",
        "Recall@1000",
        "MRR_full",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    expert_index = index_results(
        [
            ExpertResult(
                dataset=row["dataset"],
                expert=row["expert"],
                density=row["density"],
                ndcg10=row["ndcg10"],
                recall10=row.get("recall10"),
                mrr=row.get("mrr"),
                source_file=row.get("source_file"),
            )
            for row in report["experts"]
        ]
    )

    lines = [
        "# DTS-v1 Results",
        "",
        "**Stage:** V1a aggregate diagnostic",
        "",
        "This report is generated from released aggregate JSON files. It is not yet a real per-user reranking selector.",
        "",
        "## Best Aggregate Policy",
        "",
        "| Method | Mean NDCG@10 | Mean Regret vs Oracle | Win-rate vs Best Fixed |",
        "|---|---:|---:|---:|",
    ]
    best = report["best_policy"]
    lines.append(
        f"| {best['method']} | {best['mean_ndcg10']:.6f} | "
        f"{best['mean_regret_vs_oracle']:.6f} | {best['win_rate_vs_best_fixed']:.2f} |"
    )

    lines.extend(
        [
            "",
            "## Expert Table",
            "",
            "| Dataset | Density | M7 | R1 | R1-plus | R2 | R3 | Oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for dataset, expert_rows in sorted(expert_index.items(), key=lambda item: -next(iter(item[1].values())).density):
        density = next(iter(expert_rows.values())).density
        oracle = max(expert_rows.values(), key=lambda item: item.ndcg10)
        lines.append(
            f"| {dataset} | {density:.0f} | "
            f"{_fmt_expert(expert_rows, 'M7')} | "
            f"{_fmt_expert(expert_rows, 'R1')} | "
            f"{_fmt_expert(expert_rows, 'R1-plus')} | "
            f"{_fmt_expert(expert_rows, 'R2')} | "
            f"{_fmt_expert(expert_rows, 'R3')} | "
            f"{oracle.expert} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_to_dict(summary: PolicySummary) -> dict[str, Any]:
    return {
        "method": summary.method,
        "mean_ndcg10": summary.mean_ndcg10,
        "mean_regret_vs_oracle": summary.mean_regret_vs_oracle,
        "win_rate_vs_best_fixed": summary.win_rate_vs_best_fixed,
        "decisions": [asdict(decision) for decision in summary.decisions],
    }


def _fmt_expert(expert_rows: dict[str, ExpertResult], expert: str) -> str:
    row = expert_rows.get(expert)
    if row is None:
        return "-"
    return f"{row.ndcg10:.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
