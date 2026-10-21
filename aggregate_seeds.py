"""Aggregate per-episode SB3 evaluation CSVs across training seeds."""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(
    r"^(?P<condition>.+)_s(?P<seed>\d+)__(?P<env>.+)__eval(?P<episodes>\d+)\.csv$"
)
RETURN_COLUMNS = ("return", "episode_return", "reward", "episode_reward")
LENGTH_COLUMNS = ("length", "episode_length")


def pick(fieldnames, candidates, path, required=True):
    for name in candidates:
        if name in fieldnames:
            return name
    if required:
        raise KeyError(f"{path.name}: none of {candidates} in {fieldnames}")
    return None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluations across seeds.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-limit", type=int, default=500)
    args = parser.parse_args()

    per_seed = []
    for path in sorted(Path(args.eval_dir).glob("*.csv")):
        match = NAME_RE.match(path.name)
        if match is None:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ret_col = pick(reader.fieldnames, RETURN_COLUMNS, path)
            len_col = pick(reader.fieldnames, LENGTH_COLUMNS, path, required=False)
            records = list(reader)
        returns = [float(r[ret_col]) for r in records]
        lengths = [float(r[len_col]) for r in records] if len_col else []
        per_seed.append({
            "condition": match["condition"],
            "seed": int(match["seed"]),
            "env": match["env"],
            "episodes": len(returns),
            "mean_return": statistics.fmean(returns),
            "episode_std_return": statistics.pstdev(returns),
            "mean_length": statistics.fmean(lengths) if lengths else "",
            "episodes_at_time_limit": (
                sum(l >= args.time_limit for l in lengths) if lengths else ""
            ),
        })

    if not per_seed:
        raise SystemExit(f"No CSVs matching {NAME_RE.pattern} in {args.eval_dir}")

    groups = defaultdict(list)
    for row in per_seed:
        groups[(row["condition"], row["env"])].append(row)

    summary = []
    for (condition, env), rows in sorted(groups.items()):
        means = [r["mean_return"] for r in rows]
        summary.append({
            "condition": condition,
            "env": env,
            "n_seeds": len(rows),
            "seeds": " ".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
            "mean_of_seed_means": statistics.fmean(means),
            "std_across_seeds": statistics.stdev(means) if len(means) > 1 else "",
            "min_seed_mean": min(means),
            "max_seed_mean": max(means),
        })

    out = Path(args.output_dir)
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "summary.csv", summary)
    print(f"Wrote {len(per_seed)} per-seed rows and {len(summary)} summary rows to {out}")


if __name__ == "__main__":
    main()