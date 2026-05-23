#!/usr/bin/env python3
"""
Analyze human evaluation results.

Computes:
  1. Fleiss' kappa per axis (inter-annotator agreement)
  2. Mean ± std per axis
  3. ≥4 rate per axis
  4. Mood pairwise accuracy
  5. One-way ANOVA per axis by genre
  6. Score distribution data

Usage:
    python scripts/analyze_human_eval.py \
        --a1 human_eval/annotator1.csv \
        --a2 human_eval/annotator2.csv \
        --a3 human_eval/annotator3.csv \
        --mood human_eval/mood_annotated.csv \
        --profiles human_eval/main_500.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


AXES = [
    "thematic_accuracy",
    "discriminativeness",
    "rule_compliance",
    "factual_consistency",
    "coherence_fluency",
]

MOOD_AXES = [
    "dark_light", "serious_playful", "slow_fast",
    "cerebral_visceral", "bleak_hopeful", "grounded_fantastical",
    "tense_relaxed", "conventional_experimental",
    "intimate_epic", "nostalgic_contemporary",
]

AXIS_DISPLAY = {
    "thematic_accuracy": "Thematic Accuracy",
    "discriminativeness": "Discriminativeness",
    "rule_compliance": "Rule Compliance",
    "factual_consistency": "Factual Consistency",
    "coherence_fluency": "Coherence & Fluency",
}


def load_annotations(path: Path) -> list:
    """Load annotator CSV into list of dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fleiss_kappa(ratings_matrix: np.ndarray, n_categories: int = 5) -> float:
    """
    Compute Fleiss' kappa for inter-rater reliability.

    Args:
        ratings_matrix: (n_subjects, n_raters) array of category assignments (1-based)
        n_categories: number of categories (default 5 for Likert scale)

    Returns:
        Fleiss' kappa value
    """
    n_subjects, n_raters = ratings_matrix.shape

    # Build category count matrix: (n_subjects, n_categories)
    cat_counts = np.zeros((n_subjects, n_categories))
    for i in range(n_subjects):
        for j in range(n_raters):
            cat = int(ratings_matrix[i, j]) - 1  # Convert to 0-based
            if 0 <= cat < n_categories:
                cat_counts[i, cat] += 1

    # Proportion of assignments to each category
    p_j = cat_counts.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    P_i = (np.sum(cat_counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Expected agreement
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0  # Perfect agreement

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return kappa


def compute_agreement(annotations: list, axis: str) -> dict:
    """Compute Fleiss' kappa and descriptive stats for one axis."""
    # Build ratings matrix: (n_items, 3_raters)
    n_items = len(annotations[0])
    n_raters = len(annotations)

    ratings = np.zeros((n_items, n_raters))
    for j, ann in enumerate(annotations):
        for i, row in enumerate(ann):
            val = row.get(axis, "")
            if val and val.strip():
                ratings[i, j] = int(val)
            else:
                ratings[i, j] = 0  # Missing

    # Filter out items with missing ratings
    valid_mask = np.all(ratings > 0, axis=1)
    ratings_valid = ratings[valid_mask]

    if len(ratings_valid) == 0:
        return {"kappa": 0.0, "mean": 0.0, "std": 0.0, "ge4_rate": 0.0, "n": 0}

    kappa = fleiss_kappa(ratings_valid, n_categories=5)

    # Descriptive stats (mean across annotators per item, then across items)
    means_per_item = ratings_valid.mean(axis=1)
    overall_mean = means_per_item.mean()
    overall_std = means_per_item.std()

    # ≥4 rate: fraction of items where mean rating ≥ 4.0
    ge4_rate = (means_per_item >= 4.0).mean()

    return {
        "kappa": kappa,
        "mean": overall_mean,
        "std": overall_std,
        "ge4_rate": ge4_rate,
        "n": len(ratings_valid),
    }


def compute_mood_accuracy(mood_file: Path, answer_key_file: Path) -> dict:
    """Compute mood pairwise comparison accuracy."""
    with open(mood_file, encoding="utf-8") as f:
        annotated = list(csv.DictReader(f))

    with open(answer_key_file, encoding="utf-8") as f:
        answer_key = list(csv.DictReader(f))

    # Build answer key lookup
    key_by_id = {int(row["pair_id"]): row for row in answer_key}

    results_per_axis = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in annotated:
        pair_id = int(row["pair_id"])
        key = key_by_id.get(pair_id)
        if not key:
            continue

        for axis in MOOD_AXES:
            annotator_answer = row.get(f"higher_{axis}", "").strip().upper()
            if not annotator_answer or annotator_answer == "TIE":
                continue

            # Determine LLM ordering
            a_val = float(key[f"a_{axis}"])
            b_val = float(key[f"b_{axis}"])

            if abs(a_val - b_val) < 0.05:
                continue  # Too close, skip

            llm_answer = "A" if a_val > b_val else "B"

            results_per_axis[axis]["total"] += 1
            if annotator_answer == llm_answer:
                results_per_axis[axis]["correct"] += 1

    accuracies = {}
    for axis in MOOD_AXES:
        r = results_per_axis[axis]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0.0
        accuracies[axis] = {"accuracy": acc, "n": r["total"]}

    overall_correct = sum(r["correct"] for r in results_per_axis.values())
    overall_total = sum(r["total"] for r in results_per_axis.values())
    accuracies["overall"] = {
        "accuracy": overall_correct / overall_total if overall_total > 0 else 0.0,
        "n": overall_total,
    }

    return accuracies


def print_results(axis_results: dict, mood_results: dict = None):
    """Pretty-print results in paper-ready format."""
    print("\n" + "=" * 80)
    print("HUMAN EVALUATION RESULTS")
    print("=" * 80)

    print(f"\n{'Axis':<25} {'Mean±Std':>12} {'Fleiss κ':>10} {'≥4 Rate':>10} {'n':>6}")
    print("-" * 70)
    for axis in AXES:
        r = axis_results[axis]
        print(
            f"{AXIS_DISPLAY[axis]:<25} "
            f"{r['mean']:.2f}±{r['std']:.2f}  "
            f"{r['kappa']:.3f}     "
            f"{r['ge4_rate']*100:.1f}%    "
            f"{r['n']:>5}"
        )

    if mood_results:
        print(f"\n{'Mood Axis':<30} {'Accuracy':>10} {'n':>6}")
        print("-" * 50)
        for axis in MOOD_AXES:
            r = mood_results.get(axis, {"accuracy": 0, "n": 0})
            print(f"{axis:<30} {r['accuracy']*100:.1f}%    {r['n']:>5}")
        overall = mood_results.get("overall", {"accuracy": 0, "n": 0})
        print(f"{'OVERALL':<30} {overall['accuracy']*100:.1f}%    {overall['n']:>5}")

    # LaTeX-ready output
    print("\n" + "=" * 80)
    print("LATEX TABLE (copy into paper Table 3)")
    print("=" * 80)
    for axis in AXES:
        r = axis_results[axis]
        print(
            f"{AXIS_DISPLAY[axis]} & "
            f"${r['mean']:.2f} \\pm {r['std']:.2f}$ & "
            f"${r['kappa']:.2f}$ & "
            f"${r['ge4_rate']*100:.1f}\\%$ \\\\"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation")
    parser.add_argument("--a1", type=str, required=True, help="Annotator 1 CSV")
    parser.add_argument("--a2", type=str, required=True, help="Annotator 2 CSV")
    parser.add_argument("--a3", type=str, required=True, help="Annotator 3 CSV")
    parser.add_argument("--mood", type=str, default=None, help="Mood pairwise CSV (annotated)")
    parser.add_argument("--mood-key", type=str, default="human_eval/mood_pairs_answer_key.csv")
    parser.add_argument("--profiles", type=str, default="human_eval/main_500.csv")
    args = parser.parse_args()

    # Load annotations
    a1 = load_annotations(Path(args.a1))
    a2 = load_annotations(Path(args.a2))
    a3 = load_annotations(Path(args.a3))
    annotations = [a1, a2, a3]

    print(f"Loaded annotations: {len(a1)} items × 3 annotators")

    # Compute per-axis metrics
    axis_results = {}
    for axis in AXES:
        axis_results[axis] = compute_agreement(annotations, axis)

    # Mood pairwise
    mood_results = None
    if args.mood:
        mood_results = compute_mood_accuracy(Path(args.mood), Path(args.mood_key))

    # Print
    print_results(axis_results, mood_results)

    # Save to JSON
    output = {
        "axes": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in axis_results.items()},
    }
    if mood_results:
        output["mood"] = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in mood_results.items()
        }

    out_path = Path("human_eval/results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
