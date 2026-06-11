"""Run the reproducible HARL HAD3QN Overcooked experiment matrix."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


DEFAULT_LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]
DEFAULT_SEEDS = [0, 1, 2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and evaluate the HARL HAD3QN Overcooked matrix."
    )
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-env-steps", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default):
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def matching_completed_run(output_dir, layout, exp_name, seed, num_env_steps):
    root = output_dir / "overcooked" / layout / "had3qn" / exp_name
    candidates = sorted(root.glob(f"seed-{seed:05d}-*"), reverse=True)
    for run_dir in candidates:
        status = load_json(run_dir / "training_status.json", {})
        config = load_json(run_dir / "config.json", {})
        configured_steps = (
            config.get("algo_args", {}).get("train", {}).get("num_env_steps")
        )
        if (
            status.get("status") == "completed"
            and configured_steps == num_env_steps
            and (run_dir / "best_models").is_dir()
        ):
            return run_dir
    return None


def newest_run(output_dir, layout, exp_name, seed):
    root = output_dir / "overcooked" / layout / "had3qn" / exp_name
    candidates = list(root.glob(f"seed-{seed:05d}-*"))
    if not candidates:
        raise FileNotFoundError(f"Training produced no run directory under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def training_command(args, layout, seed, exp_name, num_env_steps):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "train.py"),
        "--algo",
        "had3qn",
        "--env",
        "overcooked",
        "--exp_name",
        exp_name,
        "--layout_name",
        layout,
        "--seed",
        str(seed),
        "--num_env_steps",
        str(num_env_steps),
        "--log_dir",
        str(args.output_dir.resolve()),
    ]
    if args.device == "cpu":
        command += ["--cuda", "False"]
    elif args.device == "cuda":
        command += ["--cuda", "True"]
    if args.smoke_test:
        command += [
            "--n_rollout_threads",
            "1",
            "--warmup_steps",
            "40",
            "--train_interval",
            "10",
            "--eval_interval",
            "20",
            "--buffer_size",
            "1000",
            "--batch_size",
            "32",
            "--n_eval_rollout_threads",
            "1",
            "--eval_episodes",
            "2",
            "--horizon",
            "40",
        ]
    return command


def evaluation_command(args, run_dir):
    return [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "evaluate_overcooked_had3qn.py"),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        "best",
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--overwrite",
    ]


def evaluation_is_current(args, run_dir):
    evaluation = load_json(run_dir / "evaluation.json", {})
    return (
        evaluation.get("checkpoint") == "best"
        and evaluation.get("episodes") == args.episodes
    )


def run_one(args, layout, seed, exp_name, num_env_steps):
    run_dir = matching_completed_run(
        args.output_dir, layout, exp_name, seed, num_env_steps
    )
    if run_dir is None:
        command = training_command(args, layout, seed, exp_name, num_env_steps)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        run_dir = newest_run(args.output_dir, layout, exp_name, seed)
        write_json(
            run_dir / "training_status.json",
            {
                "status": "completed",
                "completed_at": utc_now(),
                "algo": "had3qn",
                "layout": layout,
                "seed": seed,
                "num_env_steps": num_env_steps,
                "command": command,
            },
        )

    if not evaluation_is_current(args, run_dir):
        subprocess.run(evaluation_command(args, run_dir), cwd=PROJECT_ROOT, check=True)
    return run_dir


def main():
    args = parse_args()
    if args.num_env_steps <= 0 or args.episodes <= 0:
        raise ValueError("Step and episode counts must be positive")
    if args.smoke_test:
        args.layouts = args.layouts[:1]
        args.seeds = args.seeds[:1]
        num_env_steps = min(args.num_env_steps, 200)
        exp_name = f"smoke_steps_{num_env_steps}"
        status_path = args.output_dir / "overcooked_had3qn_smoke_status.json"
    else:
        num_env_steps = args.num_env_steps
        exp_name = f"steps_{num_env_steps}"
        status_path = args.output_dir / "overcooked_had3qn_batch_status.json"

    status = load_json(status_path, {"runs": {}})
    status.update({"started_at": status.get("started_at", utc_now()), "status": "running"})
    write_json(status_path, status)

    failures = 0
    for layout in args.layouts:
        for seed in args.seeds:
            key = f"{layout}/seed_{seed}"
            try:
                run_dir = run_one(args, layout, seed, exp_name, num_env_steps)
                status["runs"][key] = {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "updated_at": utc_now(),
                }
            except Exception as error:
                failures += 1
                status["runs"][key] = {
                    "status": "failed",
                    "error": repr(error),
                    "updated_at": utc_now(),
                }
                if args.stop_on_error:
                    status["status"] = "failed"
                    write_json(status_path, status)
                    raise
            write_json(status_path, status)

    status["status"] = "completed" if failures == 0 else "completed_with_failures"
    status["completed_at"] = utc_now()
    write_json(status_path, status)
    if failures:
        raise SystemExit(f"{failures} experiment(s) failed; see {status_path}")


if __name__ == "__main__":
    main()
