from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from extend_research.analysis.dts_v1_data import ExpertResult
from extend_research.selectors.threshold_selector import ThresholdPolicy, TwoThresholdPolicy


@dataclass(frozen=True)
class DatasetDecision:
    dataset: str
    density: float
    method: str
    chosen_expert: str
    chosen_ndcg10: float
    oracle_expert: str
    oracle_ndcg10: float
    regret_vs_oracle: float
    best_fixed_expert: str
    best_fixed_ndcg10: float
    delta_vs_best_fixed: float


@dataclass(frozen=True)
class PolicySummary:
    method: str
    mean_ndcg10: float
    mean_regret_vs_oracle: float
    win_rate_vs_best_fixed: float
    decisions: list[DatasetDecision]


def index_results(rows: list[ExpertResult]) -> dict[str, dict[str, ExpertResult]]:
    index: dict[str, dict[str, ExpertResult]] = defaultdict(dict)
    for row in rows:
        index[row.dataset][row.expert] = row
    return dict(index)


def evaluate_fixed_experts(rows: list[ExpertResult], fixed_experts: list[str]) -> list[PolicySummary]:
    index = index_results(rows)
    summaries = []
    for expert in fixed_experts:
        decisions = []
        for dataset, expert_rows in index.items():
            if expert not in expert_rows:
                continue
            chosen = expert_rows[expert]
            oracle = max(expert_rows.values(), key=lambda item: item.ndcg10)
            best_fixed = _best_fixed(expert_rows, fixed_experts)
            decisions.append(
                _decision(
                    dataset=dataset,
                    density=chosen.density,
                    method=f"{expert}-only",
                    chosen_expert=expert,
                    chosen_ndcg10=chosen.ndcg10,
                    oracle_expert=oracle.expert,
                    oracle_ndcg10=oracle.ndcg10,
                    best_fixed_expert=best_fixed.expert,
                    best_fixed_ndcg10=best_fixed.ndcg10,
                )
            )
        summaries.append(summarize_decisions(f"{expert}-only", decisions))
    return summaries


def evaluate_threshold_policies(
    rows: list[ExpertResult],
    policies: list[ThresholdPolicy | TwoThresholdPolicy],
    fixed_experts: list[str],
) -> list[PolicySummary]:
    index = index_results(rows)
    summaries = []

    for policy in policies:
        decisions = []
        for dataset, expert_rows in index.items():
            density = next(iter(expert_rows.values())).density
            chosen_expert = policy.choose(density)
            if chosen_expert not in expert_rows:
                continue
            chosen = expert_rows[chosen_expert]
            oracle = max(expert_rows.values(), key=lambda item: item.ndcg10)
            best_fixed = _best_fixed(expert_rows, fixed_experts)
            decisions.append(
                _decision(
                    dataset=dataset,
                    density=density,
                    method=policy.name,
                    chosen_expert=chosen_expert,
                    chosen_ndcg10=chosen.ndcg10,
                    oracle_expert=oracle.expert,
                    oracle_ndcg10=oracle.ndcg10,
                    best_fixed_expert=best_fixed.expert,
                    best_fixed_ndcg10=best_fixed.ndcg10,
                )
            )
        summaries.append(summarize_decisions(policy.name, decisions))

    return summaries


def summarize_decisions(method: str, decisions: list[DatasetDecision]) -> PolicySummary:
    if not decisions:
        return PolicySummary(
            method=method,
            mean_ndcg10=0.0,
            mean_regret_vs_oracle=0.0,
            win_rate_vs_best_fixed=0.0,
            decisions=[],
        )

    mean_ndcg = sum(item.chosen_ndcg10 for item in decisions) / len(decisions)
    mean_regret = sum(item.regret_vs_oracle for item in decisions) / len(decisions)
    wins = sum(item.delta_vs_best_fixed >= 0 for item in decisions)
    return PolicySummary(
        method=method,
        mean_ndcg10=mean_ndcg,
        mean_regret_vs_oracle=mean_regret,
        win_rate_vs_best_fixed=wins / len(decisions),
        decisions=decisions,
    )


def _best_fixed(
    expert_rows: dict[str, ExpertResult],
    fixed_experts: list[str],
) -> ExpertResult:
    available = [expert_rows[expert] for expert in fixed_experts if expert in expert_rows]
    if not available:
        available = list(expert_rows.values())
    return max(available, key=lambda item: item.ndcg10)


def _decision(
    dataset: str,
    density: float,
    method: str,
    chosen_expert: str,
    chosen_ndcg10: float,
    oracle_expert: str,
    oracle_ndcg10: float,
    best_fixed_expert: str,
    best_fixed_ndcg10: float,
) -> DatasetDecision:
    return DatasetDecision(
        dataset=dataset,
        density=density,
        method=method,
        chosen_expert=chosen_expert,
        chosen_ndcg10=chosen_ndcg10,
        oracle_expert=oracle_expert,
        oracle_ndcg10=oracle_ndcg10,
        regret_vs_oracle=oracle_ndcg10 - chosen_ndcg10,
        best_fixed_expert=best_fixed_expert,
        best_fixed_ndcg10=best_fixed_ndcg10,
        delta_vs_best_fixed=chosen_ndcg10 - best_fixed_ndcg10,
    )
