"""Summarize HARL HAD3QN and PantheonRL DQN Overcooked evaluations."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harl-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--pantheon-root", type=Path, default=Path("/workspace/PantheonRL/results")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/overcooked_comparison.json")
    )
    return parser.parse_args()


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def load_config(path):
    config_path = path.parent / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def metric(result, flat_key, episode_key):
    if result.get(flat_key) is not None:
        return result[flat_key]
    return mean_or_none(result.get(episode_key, []))


def load_rows(root, source):
    rows = []
    for path in root.rglob("evaluation.json"):
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        config = load_config(path)
        if source == "harl" and result.get("algo") != "had3qn":
            continue
        if source == "pantheonrl" and result.get("algo") != "dqn":
            continue
        if "alternating_dqn" in path.parts:
            experiment = "alternating_dqn"
        elif source == "pantheonrl":
            experiment = "independent_dqn"
        else:
            experiment = "had3qn"
        if source == "harl":
            exploration_fraction = None
            epsilon = config.get("algo_args", {}).get("algo", {}).get("epsilon")
        else:
            effective = config.get("effective_hyperparameters", {})
            exploration_fraction = effective.get(
                "exploration_fraction",
                config.get("model_kwargs", {}).get("exploration_fraction"),
            )
            epsilon = effective.get("exploration_final_eps")
        rows.append(
            {
                "source": source,
                "experiment": experiment,
                "layout": result.get("layout"),
                "seed": result.get("seed"),
                "exploration_fraction": exploration_fraction,
                "epsilon": epsilon,
                "run_dir": result.get("run_dir", str(path.parent)),
                "mean_total_return": metric(
                    result, "mean_return", "episode_returns"
                ),
                "mean_sparse_return": metric(
                    result, "mean_sparse_return", "episode_sparse_returns"
                ),
                "mean_shaped_return": metric(
                    result, "mean_shaped_return", "episode_shaped_returns"
                ),
                "mean_deliveries": metric(
                    result, "mean_deliveries", "episode_deliveries"
                ),
            }
        )
    return rows


def group_rows(rows):
    metrics = [
        "mean_total_return",
        "mean_sparse_return",
        "mean_shaped_return",
        "mean_deliveries",
    ]
    grouped = {}
    for row in rows:
        key = (
            row["source"],
            row["experiment"],
            row["layout"],
            row["exploration_fraction"],
            row["epsilon"],
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group in grouped.items():
        summary = {
            "source": key[0],
            "experiment": key[1],
            "layout": key[2],
            "exploration_fraction": key[3],
            "epsilon": key[4],
            "runs": len(group),
        }
        for metric_name in metrics:
            values = [
                row[metric_name]
                for row in group
                if row[metric_name] is not None
            ]
            summary[metric_name] = mean_or_none(values)
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row["experiment"],
            str(row["layout"]),
            str(row["exploration_fraction"]),
        )
    )
    return summaries


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    rows = load_rows(args.harl_root, "harl")
    rows += load_rows(args.pantheon_root, "pantheonrl")
    rows.sort(key=lambda row: (row["experiment"], str(row["layout"]), str(row["seed"])))
    summaries = group_rows(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump({"runs": rows, "groups": summaries}, file, indent=2, sort_keys=True)
        file.write("\n")

    csv_path = args.output.with_suffix(".csv")
    summary_csv_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    write_csv(csv_path, rows)
    write_csv(summary_csv_path, summaries)
    print(
        f"Wrote {len(rows)} runs and {len(summaries)} grouped comparisons "
        f"under {args.output.parent}"
    )


if __name__ == "__main__":
    main()
