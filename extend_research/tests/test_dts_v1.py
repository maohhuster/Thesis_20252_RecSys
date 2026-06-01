from extend_research.analysis.dts_v1_data import ExpertResult
from extend_research.evaluation.router_metrics import evaluate_threshold_policies
from extend_research.selectors.threshold_selector import ThresholdPolicy, TwoThresholdPolicy


def test_threshold_policy_uses_sparse_expert_below_threshold() -> None:
    policy = ThresholdPolicy(threshold=50, sparse_expert="R1-plus")

    assert policy.choose(13) == "R1-plus"
    assert policy.choose(50) == "M7"
    assert policy.choose(1160) == "M7"


def test_threshold_policy_summary_computes_regret() -> None:
    rows = [
        ExpertResult(dataset="Dense", expert="M7", density=100, ndcg10=0.10),
        ExpertResult(dataset="Dense", expert="R1", density=100, ndcg10=0.09),
        ExpertResult(dataset="Sparse", expert="M7", density=10, ndcg10=0.05),
        ExpertResult(dataset="Sparse", expert="R1", density=10, ndcg10=0.08),
    ]
    summaries = evaluate_threshold_policies(
        rows=rows,
        policies=[ThresholdPolicy(threshold=50, sparse_expert="R1")],
        fixed_experts=["M7", "R1"],
    )

    assert len(summaries) == 1
    assert summaries[0].mean_ndcg10 == 0.09
    assert summaries[0].mean_regret_vs_oracle == 0.0
    assert summaries[0].win_rate_vs_best_fixed == 1.0


def test_two_threshold_policy_routes_low_mid_high_density() -> None:
    policy = TwoThresholdPolicy(
        name="test-2t",
        low_threshold=200,
        high_threshold=500,
        low_expert="R1",
        mid_expert="R1-plus",
        high_expert="M7",
    )

    assert policy.choose(13) == "R1"
    assert policy.choose(225) == "R1-plus"
    assert policy.choose(1160) == "M7"
