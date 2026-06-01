from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from inspect_cross_domain_steam import (
    build_metadata_text,
    extract_first,
    file_group_status,
    iter_records,
    resolve_config,
    resolve_path,
    safe_int,
)


Interaction = tuple[str, str, int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Steam normalized tables and density regimes.")
    parser.add_argument("--config", type=Path, default=Path("configs/cross_domain_steam_export.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    config = read_json(config_path)
    raw_dir = resolve_path(config["raw_dir"], config_path)
    processed_dir = resolve_path(config["processed_dir"], config_path)
    output_dir = resolve_path(config["output_dir"], config_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_status = {
        group: file_group_status(raw_dir, candidates)
        for group, candidates in config["files"].items()
    }
    require_found(file_status, ["games", "user_items"])

    print("Loading Steam item metadata", flush=True)
    items = load_items(
        file_status["games"]["path"],
        max_records=int(config["limits"]["max_game_records"]),
    )
    print(f"  items={len(items)}", flush=True)

    print("Loading Steam review timestamps", flush=True)
    review_index = load_review_index(
        file_status["reviews"]["path"],
        max_records=int(config["limits"]["max_review_records"]),
        recommended_only=bool(config["filters"]["recommended_only"]),
    )
    print(f"  review user-item pairs={len(review_index)}", flush=True)

    print("Loading Steam user-item interactions", flush=True)
    interactions = load_user_item_interactions(
        file_status["user_items"]["path"],
        review_index=review_index,
        max_records=int(config["limits"]["max_user_records"]),
        max_interactions=int(config["limits"]["max_interactions"]),
        min_playtime_forever=int(config["filters"]["min_playtime_forever"]),
    )
    print(f"  interactions={len(interactions)}", flush=True)

    base_dir = processed_dir / "base"
    write_interactions(base_dir / "interactions.csv", interactions)
    write_items_for_interactions(base_dir / "items.csv", interactions, items)

    summary_rows = []
    summary_rows.append(summary_row("base", 0, 0, interactions, items))

    for regime in config["regimes"]:
        filtered = iterative_k_core(
            interactions,
            min_user_degree=int(regime["min_user_degree"]),
            min_item_degree=int(regime["min_item_degree"]),
        )
        regime_dir = processed_dir / regime["name"]
        write_interactions(regime_dir / "interactions.csv", filtered)
        write_items_for_interactions(regime_dir / "items.csv", filtered, items)
        summary_rows.append(
            summary_row(
                regime["name"],
                int(regime["min_user_degree"]),
                int(regime["min_item_degree"]),
                filtered,
                items,
            )
        )
        print(f"  {regime['name']}: interactions={len(filtered)}", flush=True)

    write_csv(output_dir / "steam_density_summary.csv", summary_rows)
    write_json(
        output_dir / "steam_export_summary.json",
        {
            "experiment_name": config["experiment_name"],
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "file_status": printable_file_status(file_status),
            "notes": config["notes"],
            "summary": summary_rows,
        },
    )
    write_markdown(output_dir / "Cross_Domain_Steam_Density_Design.md", config, summary_rows)
    print(f"Wrote Steam export outputs to {processed_dir}", flush=True)
    print(f"Wrote Steam density-design report to {output_dir}", flush=True)


def require_found(file_status: dict[str, dict[str, Any]], groups: list[str]) -> None:
    missing = [group for group in groups if not file_status[group]["found"]]
    if missing:
        raise FileNotFoundError(f"Missing required Steam file groups: {', '.join(missing)}")


def load_items(path: Path | None, max_records: int) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    items = {}
    for record in iter_records(path, max_records=record_limit(max_records)):
        item_id = extract_first(record, ["id", "app_id", "item_id", "product_id"])
        if item_id is None:
            continue
        item_id_str = str(item_id)
        title = str(extract_first(record, ["title", "name", "app_name", "item_name"]) or "")
        row = {
            "item_id": item_id_str,
            "title_or_name": title,
            "metadata_text": build_metadata_text(record),
            "genres": json.dumps(record.get("genres", []), ensure_ascii=False),
            "tags": json.dumps(record.get("tags", []), ensure_ascii=False),
            "specs": json.dumps(record.get("specs", []), ensure_ascii=False),
            "developer": stringify(record.get("developer", "")),
            "publisher": stringify(record.get("publisher", "")),
            "release_date": stringify(record.get("release_date", "")),
            "price": stringify(record.get("price", "")),
            "raw_metadata_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
        }
        items[item_id_str] = row
    return items


def load_review_index(
    path: Path | None,
    max_records: int,
    recommended_only: bool,
) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    review_index = {}
    for record in iter_records(path, max_records=record_limit(max_records)):
        user_id = extract_first(record, ["user_id", "user"])
        reviews = record.get("reviews") if isinstance(record, dict) else None
        if user_id is None or not isinstance(reviews, list):
            continue
        for review in reviews:
            if not isinstance(review, dict):
                continue
            if recommended_only and review.get("recommend") is False:
                continue
            item_id = extract_first(review, ["item_id", "id", "app_id", "product_id"])
            if item_id is None:
                continue
            timestamp = stringify(extract_first(review, ["posted", "date", "timestamp"]) or "")
            review_index[(str(user_id), str(item_id))] = timestamp
    return review_index


def load_user_item_interactions(
    path: Path | None,
    review_index: dict[tuple[str, str], str],
    max_records: int,
    max_interactions: int,
    min_playtime_forever: int,
) -> list[Interaction]:
    if path is None:
        return []
    rows: list[Interaction] = []
    seen: set[tuple[str, str]] = set()
    for record in iter_records(path, max_records=record_limit(max_records)):
        user_id = extract_first(record, ["user_id", "user"])
        items = record.get("items") if isinstance(record, dict) else None
        if user_id is None or not isinstance(items, list):
            continue
        user_id_str = str(user_id)
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = extract_first(item, ["item_id", "id", "app_id"])
            if item_id is None:
                continue
            item_id_str = str(item_id)
            playtime = safe_int(extract_first(item, ["playtime_forever", "playtime"]))
            if playtime < min_playtime_forever:
                continue
            key = (user_id_str, item_id_str)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                (
                    user_id_str,
                    item_id_str,
                    playtime,
                    stringify(extract_first(item, ["item_name", "name", "title"]) or ""),
                    review_index.get(key, ""),
                )
            )
            if max_interactions > 0 and len(rows) >= max_interactions:
                return rows
    return rows


def iterative_k_core(
    interactions: list[Interaction],
    min_user_degree: int,
    min_item_degree: int,
) -> list[Interaction]:
    current = interactions
    while True:
        user_degree = Counter(row[0] for row in current)
        item_degree = Counter(row[1] for row in current)
        next_rows = [
            row for row in current
            if user_degree[row[0]] >= min_user_degree and item_degree[row[1]] >= min_item_degree
        ]
        if len(next_rows) == len(current):
            return next_rows
        current = next_rows


def summary_row(
    regime: str,
    min_user_degree: int,
    min_item_degree: int,
    interactions: list[Interaction],
    items: dict[str, dict[str, str]],
) -> dict[str, Any]:
    user_degree = Counter(row[0] for row in interactions)
    item_degree = Counter(row[1] for row in interactions)
    retained_item_ids = set(item_degree)
    fallback_titles = fallback_title_by_item(interactions)
    metadata_covered = sum(1 for item_id in retained_item_ids if item_id in items)
    title_covered = sum(
        1 for item_id in retained_item_ids
        if (item_id in items and items[item_id]["title_or_name"]) or fallback_titles.get(item_id)
    )
    n_users = len(user_degree)
    n_items = len(item_degree)
    n_interactions = len(interactions)
    return {
        "regime": regime,
        "min_user_degree": min_user_degree,
        "min_item_degree": min_item_degree,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density": density(n_interactions, n_users, n_items),
        "metadata_coverage": metadata_covered / n_items if n_items else 0.0,
        "title_coverage": title_covered / n_items if n_items else 0.0,
        "mean_interactions_per_user": float(np.mean(list(user_degree.values()))) if user_degree else 0.0,
        "mean_interactions_per_item": float(np.mean(list(item_degree.values()))) if item_degree else 0.0,
        "p50_item_degree": quantile(item_degree, 0.50),
        "p90_item_degree": quantile(item_degree, 0.90),
    }


def write_interactions(path: Path, interactions: list[Interaction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "user_id",
                "item_id",
                "timestamp",
                "rating_or_value",
                "interaction_source",
                "item_title",
            ],
        )
        writer.writeheader()
        for user_id, item_id, playtime, item_title, timestamp in interactions:
            writer.writerow(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "rating_or_value": playtime,
                    "interaction_source": "steam_playtime",
                    "item_title": item_title,
                }
            )


def write_items_for_interactions(
    path: Path,
    interactions: list[Interaction],
    items: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    retained_item_ids = sorted(set(row[1] for row in interactions), key=sort_key)
    fallback_titles = fallback_title_by_item(interactions)
    fieldnames = [
        "item_id",
        "title_or_name",
        "metadata_text",
        "genres",
        "tags",
        "specs",
        "developer",
        "publisher",
        "release_date",
        "price",
        "raw_metadata_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item_id in retained_item_ids:
            row = items.get(item_id)
            if row is None:
                title = fallback_titles.get(item_id, "")
                row = {
                    "item_id": item_id,
                    "title_or_name": title,
                    "metadata_text": title,
                    "genres": "[]",
                    "tags": "[]",
                    "specs": "[]",
                    "developer": "",
                    "publisher": "",
                    "release_date": "",
                    "price": "",
                    "raw_metadata_json": "{}",
                }
            elif not row["title_or_name"] and fallback_titles.get(item_id):
                row = dict(row)
                row["title_or_name"] = fallback_titles[item_id]
                if not row["metadata_text"]:
                    row["metadata_text"] = fallback_titles[item_id]
            writer.writerow(row)


def write_markdown(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Domain Steam Density Design",
        "",
        "Steam raw data was exported into normalized CSV tables for the base set and",
        "three iterative k-core density regimes.",
        "",
        "Important protocol note: user-item playtime records do not include real",
        "timestamps. Review dates are attached only for matching user-item pairs. A",
        "future train/validation/test split must either use the review subset for",
        "temporal evaluation or define an explicit random/user-level split policy.",
        "",
        f"Default interaction cap: `{config['limits']['max_interactions']}`",
        "",
        "## Density Summary",
        "",
        "| Regime | k-user | k-item | Users | Items | Interactions | Density | Metadata Coverage | Title Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['regime']} | {row['min_user_degree']} | {row['min_item_degree']} | "
            f"{row['n_users']} | {row['n_items']} | {row['n_interactions']} | "
            f"{row['density']:.8f} | {row['metadata_coverage']:.6f} | "
            f"{row['title_coverage']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            "Generated files are under `data/processed/cross_domain/steam/` and are",
            "gitignored.",
            "",
            "Each regime contains:",
            "",
            "- `interactions.csv`",
            "- `items.csv`",
            "",
            "## Next Step",
            "",
            "Use this export to define the first Steam cross-domain matrix smoke run.",
            "Before training any model, choose the split policy explicitly because the",
            "main playtime source lacks timestamps.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def printable_file_status(file_status: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for group, status in file_status.items():
        out[group] = {
            "found": status["found"],
            "selected": status["selected"],
            "path": str(status["path"]) if status["path"] else "",
        }
    return out


def fallback_title_by_item(interactions: list[Interaction]) -> dict[str, str]:
    titles = {}
    for _user_id, item_id, _playtime, item_title, _timestamp in interactions:
        if item_title and item_id not in titles:
            titles[item_id] = item_title
    return titles


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def density(n_interactions: int, n_users: int, n_items: int) -> float:
    denominator = n_users * n_items
    return float(n_interactions / denominator) if denominator else 0.0


def quantile(counter: Counter[str], q: float) -> float:
    if not counter:
        return 0.0
    return float(np.quantile(np.array(list(counter.values()), dtype=np.float32), q))


def sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)


def record_limit(value: int) -> int:
    return 1_000_000_000 if value <= 0 else value


if __name__ == "__main__":
    main()
