#!/usr/bin/env python3
"""Run R2 (KAR — Knowledge-Augmented Recommendation, MoE adapter) on Amazon-Books.

Architecture: LightGCN backbone + 4-expert hybrid adapter consuming
LLM profile + mood features (1034-dim).

Examples:
    python scripts/run_amazon_r2.py
    python scripts/run_amazon_r2.py --seed 123 --gpu 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _amazon_run_helper import run_amazon_config


if __name__ == "__main__":
    run_amazon_config(model="kar", features="llm_prof_mood", label="R2")
