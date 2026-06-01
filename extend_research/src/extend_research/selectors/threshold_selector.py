from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdPolicy:
    threshold: float
    sparse_expert: str
    dense_expert: str = "M7"

    @property
    def name(self) -> str:
        return f"DTS-hard-{self.sparse_expert}-T{_format_threshold(self.threshold)}"

    def choose(self, density: float) -> str:
        if density < self.threshold:
            return self.sparse_expert
        return self.dense_expert


@dataclass(frozen=True)
class TwoThresholdPolicy:
    name: str
    low_threshold: float
    high_threshold: float
    low_expert: str
    mid_expert: str
    high_expert: str = "M7"

    def choose(self, density: float) -> str:
        if density < self.low_threshold:
            return self.low_expert
        if density < self.high_threshold:
            return self.mid_expert
        return self.high_expert


def build_threshold_policies(
    thresholds: list[float],
    sparse_experts: list[str],
    dense_expert: str = "M7",
) -> list[ThresholdPolicy]:
    return [
        ThresholdPolicy(threshold=float(threshold), sparse_expert=expert, dense_expert=dense_expert)
        for expert in sparse_experts
        for threshold in thresholds
    ]


def build_two_threshold_policies(policy_configs: list[dict]) -> list[TwoThresholdPolicy]:
    return [
        TwoThresholdPolicy(
            name=str(item["name"]),
            low_threshold=float(item["low_threshold"]),
            high_threshold=float(item["high_threshold"]),
            low_expert=str(item["low_expert"]),
            mid_expert=str(item["mid_expert"]),
            high_expert=str(item.get("high_expert", "M7")),
        )
        for item in policy_configs
    ]


def _format_threshold(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")
