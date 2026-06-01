from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Steam cross-domain raw data readiness.")
    parser.add_argument("--config", type=Path, default=Path("configs/cross_domain_steam_readiness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    config = read_json(config_path)
    raw_dir = resolve_path(config["raw_dir"], config_path)
    output_dir = resolve_path(config["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_status = {
        group: file_group_status(raw_dir, candidates)
        for group, candidates in config["files"].items()
    }

    games_stats = inspect_games(
        file_status["games"]["path"],
        max_records=int(config["limits"]["max_game_records"]),
    )
    interaction_payload = inspect_interactions(
        user_items_path=file_status["user_items"]["path"],
        reviews_path=file_status["reviews"]["path"],
        config=config,
    )
    readiness = build_readiness(config, raw_dir, file_status, games_stats, interaction_payload)

    write_json(output_dir / "steam_readiness_summary.json", readiness)
    write_csv(output_dir / "steam_sample_interactions.csv", interaction_payload["sample_rows"])
    write_markdown(output_dir / "Cross_Domain_Steam_Smoke.md", readiness)
    print(f"Wrote Steam readiness outputs to {output_dir}", flush=True)
    print(f"Status: {readiness['status']}", flush=True)


def file_group_status(raw_dir: Path, candidates: list[str]) -> dict[str, Any]:
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return {"found": True, "path": path, "selected": name, "candidates": candidates}
    return {"found": False, "path": None, "selected": "", "candidates": candidates}


def inspect_games(path: Path | None, max_records: int) -> dict[str, Any]:
    if path is None:
        return {
            "found": False,
            "records_scanned": 0,
            "unique_items": 0,
            "title_coverage": 0.0,
            "metadata_text_coverage": 0.0,
        }

    item_ids = set()
    title_count = 0
    metadata_count = 0
    records_scanned = 0
    for record in iter_records(path, max_records=max_records):
        records_scanned += 1
        item_id = extract_first(record, ["id", "app_id", "item_id", "product_id"])
        if item_id is not None:
            item_ids.add(str(item_id))
        title = extract_first(record, ["title", "name", "app_name", "item_name"])
        if title:
            title_count += 1
        metadata_text = build_metadata_text(record)
        if metadata_text:
            metadata_count += 1

    denominator = max(records_scanned, 1)
    return {
        "found": True,
        "path": str(path),
        "records_scanned": records_scanned,
        "unique_items": len(item_ids),
        "title_coverage": title_count / denominator,
        "metadata_text_coverage": metadata_count / denominator,
    }


def inspect_interactions(
    user_items_path: Path | None,
    reviews_path: Path | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    user_degree: Counter[str] = Counter()
    item_degree: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    max_interactions = int(config["limits"]["max_interactions"])
    max_sample_rows = int(config["limits"]["max_sample_rows"])

    for row in iter_user_item_interactions(
        user_items_path,
        max_records=int(config["limits"]["max_user_records"]),
        min_playtime_forever=int(config["filters"]["min_playtime_forever"]),
    ):
        add_interaction(row, user_degree, item_degree, sample_rows, source_counts, max_sample_rows)
        if sum(source_counts.values()) >= max_interactions:
            break

    if sum(source_counts.values()) < max_interactions:
        for row in iter_review_interactions(
            reviews_path,
            max_records=int(config["limits"]["max_review_records"]),
            recommended_only=bool(config["filters"]["recommended_only"]),
        ):
            add_interaction(row, user_degree, item_degree, sample_rows, source_counts, max_sample_rows)
            if sum(source_counts.values()) >= max_interactions:
                break

    n_interactions = int(sum(source_counts.values()))
    return {
        "source_counts": dict(source_counts),
        "n_interactions": n_interactions,
        "n_users": len(user_degree),
        "n_items": len(item_degree),
        "density": density(n_interactions, len(user_degree), len(item_degree)),
        "interactions_per_user": summarize_counts(user_degree),
        "interactions_per_item": summarize_counts(item_degree),
        "k_core_estimates": k_core_estimates(user_degree, item_degree, config["density"]["k_core_levels"]),
        "sample_rows": sample_rows,
    }


def add_interaction(
    row: dict[str, Any],
    user_degree: Counter[str],
    item_degree: Counter[str],
    sample_rows: list[dict[str, Any]],
    source_counts: Counter[str],
    max_sample_rows: int,
) -> None:
    user_id = str(row["user_id"])
    item_id = str(row["item_id"])
    user_degree[user_id] += 1
    item_degree[item_id] += 1
    source_counts[str(row["source"])] += 1
    if len(sample_rows) < max_sample_rows:
        sample_rows.append(row)


def iter_user_item_interactions(
    path: Path | None,
    max_records: int,
    min_playtime_forever: int,
) -> Iterable[dict[str, Any]]:
    if path is None:
        return
    for record in iter_records(path, max_records=max_records):
        user_id = extract_first(record, ["user_id", "user"])
        items = record.get("items") if isinstance(record, dict) else None
        if user_id is None or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = extract_first(item, ["item_id", "id", "app_id"])
            if item_id is None:
                continue
            playtime = safe_int(extract_first(item, ["playtime_forever", "playtime"]))
            if playtime < min_playtime_forever:
                continue
            yield {
                "source": "user_items",
                "user_id": str(user_id),
                "item_id": str(item_id),
                "timestamp": "",
                "interaction_value": playtime,
                "item_title": str(extract_first(item, ["item_name", "name", "title"]) or ""),
            }


def iter_review_interactions(
    path: Path | None,
    max_records: int,
    recommended_only: bool,
) -> Iterable[dict[str, Any]]:
    if path is None:
        return
    for record in iter_records(path, max_records=max_records):
        if "reviews" in record and isinstance(record["reviews"], list):
            user_id = extract_first(record, ["user_id", "user"])
            for review in record["reviews"]:
                if not isinstance(review, dict):
                    continue
                row = review_interaction(user_id, review, recommended_only)
                if row:
                    yield row
            continue

        user_id = extract_first(record, ["user_id", "user", "author_id"])
        row = review_interaction(user_id, record, recommended_only)
        if row:
            yield row


def review_interaction(
    user_id: Any,
    review: dict[str, Any],
    recommended_only: bool,
) -> dict[str, Any] | None:
    item_id = extract_first(review, ["item_id", "id", "app_id", "product_id"])
    if user_id is None or item_id is None:
        return None
    recommended = extract_first(review, ["recommend", "recommended", "voted_up"])
    if recommended_only and recommended is False:
        return None
    return {
        "source": "reviews",
        "user_id": str(user_id),
        "item_id": str(item_id),
        "timestamp": str(extract_first(review, ["posted", "date", "timestamp"]) or ""),
        "interaction_value": 1 if recommended is not False else 0,
        "item_title": str(extract_first(review, ["item_name", "name", "title"]) or ""),
    }


def iter_records(path: Path, max_records: int) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            record = parse_record(line)
            if isinstance(record, dict):
                yield record


def parse_record(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (SyntaxError, ValueError):
            return None


def build_metadata_text(record: dict[str, Any]) -> str:
    fields = [
        extract_first(record, ["title", "name", "app_name", "item_name"]),
        extract_first(record, ["genres", "genre"]),
        extract_first(record, ["tags", "popular_tags"]),
        extract_first(record, ["specs", "categories"]),
        extract_first(record, ["developer", "publisher"]),
    ]
    parts = []
    for value in fields:
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif value:
            parts.append(str(value))
    return " ".join(parts).strip()


def build_readiness(
    config: dict[str, Any],
    raw_dir: Path,
    file_status: dict[str, dict[str, Any]],
    games_stats: dict[str, Any],
    interaction_payload: dict[str, Any],
) -> dict[str, Any]:
    required_groups = ["games", "user_items"]
    missing_required = [group for group in required_groups if not file_status[group]["found"]]
    has_interactions = interaction_payload["n_interactions"] > 0
    status = "ready_for_density_design" if not missing_required and has_interactions else "missing_raw_data"
    if not missing_required and not has_interactions:
        status = "raw_data_found_but_no_interactions"

    return {
        "experiment_name": config["experiment_name"],
        "status": status,
        "raw_dir": str(raw_dir),
        "source_urls": config["source_urls"],
        "file_status": printable_file_status(file_status),
        "missing_required_groups": missing_required,
        "games": games_stats,
        "interactions": {
            key: value
            for key, value in interaction_payload.items()
            if key != "sample_rows"
        },
        "recommendation": recommendation(status, missing_required),
    }


def printable_file_status(file_status: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for group, status in file_status.items():
        out[group] = {
            "found": status["found"],
            "selected": status["selected"],
            "path": str(status["path"]) if status["path"] else "",
            "candidates": status["candidates"],
        }
    return out


def recommendation(status: str, missing_required: list[str]) -> str:
    if status == "ready_for_density_design":
        return "Proceed to define Steam density levels and temporal split."
    if "games" in missing_required and "user_items" in missing_required:
        return "Download Steam game metadata and user/item interaction files before rerunning."
    if "games" in missing_required:
        return "Add Steam game metadata before building item content/profile features."
    if "user_items" in missing_required:
        return "Add Steam user/item interaction data before density analysis."
    return "Raw files were found, but no usable interactions passed the smoke filters."


def summarize_counts(counter: Counter[str]) -> dict[str, float]:
    if not counter:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    values = np.array(list(counter.values()), dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def k_core_estimates(
    user_degree: Counter[str],
    item_degree: Counter[str],
    levels: list[int],
) -> list[dict[str, Any]]:
    out = []
    for level in levels:
        out.append(
            {
                "k": int(level),
                "users_with_degree_at_least_k": sum(1 for value in user_degree.values() if value >= level),
                "items_with_degree_at_least_k": sum(1 for value in item_degree.values() if value >= level),
            }
        )
    return out


def density(n_interactions: int, n_users: int, n_items: int) -> float:
    denominator = n_users * n_items
    return float(n_interactions / denominator) if denominator else 0.0


def extract_first(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, readiness: dict[str, Any]) -> None:
    interactions = readiness["interactions"]
    lines = [
        "# Cross-Domain Steam Smoke Report",
        "",
        f"Status: **{readiness['status']}**",
        "",
        "## File Status",
        "",
        "| Group | Found | Selected |",
        "|---|---|---|",
    ]
    for group, status in readiness["file_status"].items():
        lines.append(f"| {group} | {status['found']} | `{status['selected']}` |")

    lines.extend(
        [
            "",
            "## Interaction Summary",
            "",
            f"- users: {interactions['n_users']}",
            f"- items: {interactions['n_items']}",
            f"- interactions: {interactions['n_interactions']}",
            f"- density: {interactions['density']:.8f}",
            f"- source counts: `{json.dumps(interactions['source_counts'])}`",
            "",
            "## Recommendation",
            "",
            readiness["recommendation"],
            "",
            "## Source URLs",
            "",
        ]
    )
    for name, url in readiness["source_urls"].items():
        lines.append(f"- {name}: <{url}>")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_config(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def resolve_path(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent.parent / candidate).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
