from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Return a small summary object for experiment orchestration checks."""
    output_dir = Path(config.get("output_dir", "results"))
    training = config.get("training", {})

    return {
        "experiment_name": config.get("experiment_name", "unnamed"),
        "seed": config.get("seed", 42),
        "epochs": training.get("epochs", 0),
        "output_dir": str(output_dir),
    }
