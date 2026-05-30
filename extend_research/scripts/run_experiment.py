from __future__ import annotations

import argparse
import json
from pathlib import Path

from extend_research import load_config
from extend_research.experiment import summarize_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an extension research experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to a JSON experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = summarize_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
