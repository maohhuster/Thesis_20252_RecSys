from extend_research.config import load_config
from extend_research.experiment import summarize_experiment


def test_load_default_config() -> None:
    config = load_config("configs/default.json")

    assert config["experiment_name"] == "baseline_extension"


def test_summarize_experiment() -> None:
    summary = summarize_experiment(
        {
            "experiment_name": "unit_test",
            "seed": 7,
            "training": {"epochs": 3},
            "output_dir": "results/test",
        }
    )

    assert summary == {
        "experiment_name": "unit_test",
        "seed": 7,
        "epochs": 3,
        "output_dir": "results/test",
    }
