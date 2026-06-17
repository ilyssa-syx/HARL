"""Run the fair-comparison HARL HAD3QN Overcooked experiment matrix."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


DEFAULT_LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TOTAL_ENV_STEPS = 500_000
DEFAULT_WARMUP_STEPS = 50_000
DEFAULT_GAMMA = 0.99
DEFAULT_EPSILON = 0.05
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and evaluate the HARL HAD3QN Overcooked matrix."
    )
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=DEFAULT_TOTAL_ENV_STEPS,
        help="Total environment interactions, including random warmup.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Random replay-buffer warmup interactions included in the total budget.",
    )
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


def matching_completed_run(
    output_dir,
    layout,
    exp_name,
    seed,
    total_env_steps,
    warmup_steps,
    training_env_steps,
):
    root = output_dir / "overcooked" / layout / "had3qn" / exp_name
    candidates = sorted(root.glob(f"seed-{seed:05d}-*"), reverse=True)
    for run_dir in candidates:
        status = load_json(run_dir / "training_status.json", {})
        config = load_json(run_dir / "config.json", {})
        algo_args = config.get("algo_args", {})
        train_args = algo_args.get("train", {})
        had3qn_args = algo_args.get("algo", {})
        if (
            status.get("status") == "completed"
            and train_args.get("num_env_steps") == training_env_steps
            and train_args.get("warmup_steps") == warmup_steps
            and training_env_steps + warmup_steps == total_env_steps
            and had3qn_args.get("gamma") == DEFAULT_GAMMA
            and had3qn_args.get("epsilon") == DEFAULT_EPSILON
            and algo_args.get("eval", {}).get("use_eval") is False
            and (run_dir / "models").is_dir()
        ):
            return run_dir
    return None


def newest_run(output_dir, layout, exp_name, seed):
    root = output_dir / "overcooked" / layout / "had3qn" / exp_name
    candidates = list(root.glob(f"seed-{seed:05d}-*"))
    if not candidates:
        raise FileNotFoundError(f"Training produced no run directory under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def training_command(args, layout, seed, exp_name, training_env_steps, warmup_steps):
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
        str(training_env_steps),
        "--warmup_steps",
        str(warmup_steps),
        "--gamma",
        str(DEFAULT_GAMMA),
        "--epsilon",
        str(DEFAULT_EPSILON),
        "--use_eval",
        "False",
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
            "--train_interval",
            "10",
            "--buffer_size",
            "1000",
            "--batch_size",
            "32",
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
        "final",
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--overwrite",
    ]


def evaluation_is_current(args, run_dir):
    evaluation = load_json(run_dir / "evaluation.json", {})
    return (
        evaluation.get("checkpoint") == "final"
        and evaluation.get("episodes") == args.episodes
    )


def run_one(
    args,
    layout,
    seed,
    exp_name,
    total_env_steps,
    warmup_steps,
    training_env_steps,
):
    run_dir = matching_completed_run(
        args.output_dir,
        layout,
        exp_name,
        seed,
        total_env_steps,
        warmup_steps,
        training_env_steps,
    )
    if run_dir is None:
        command = training_command(
            args, layout, seed, exp_name, training_env_steps, warmup_steps
        )
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
                "total_env_steps": total_env_steps,
                "warmup_steps": warmup_steps,
                "training_env_steps": training_env_steps,
                "gamma": DEFAULT_GAMMA,
                "epsilon": DEFAULT_EPSILON,
                "checkpoint": "final",
                "command": command,
            },
        )

    if not evaluation_is_current(args, run_dir):
        subprocess.run(evaluation_command(args, run_dir), cwd=PROJECT_ROOT, check=True)
    return run_dir


def main():
    args = parse_args()
    if args.total_env_steps <= 0 or args.episodes <= 0:
        raise ValueError("Total steps and episode counts must be positive")
    if args.warmup_steps < 0:
        raise ValueError("Warmup steps must be non-negative")
    if args.warmup_steps >= args.total_env_steps:
        raise ValueError("Warmup steps must be less than total environment steps")

    if args.smoke_test:
        args.layouts = args.layouts[:1]
        args.seeds = args.seeds[:1]
        total_env_steps = min(args.total_env_steps, 200)
        warmup_steps = min(args.warmup_steps, 40)
        exp_name = f"fair_smoke_total_steps_{total_env_steps}"
        status_path = args.output_dir / "overcooked_had3qn_fair_smoke_status.json"
    else:
        total_env_steps = args.total_env_steps
        warmup_steps = args.warmup_steps
        exp_name = f"fair_total_steps_{total_env_steps}"
        status_path = args.output_dir / "overcooked_had3qn_fair_batch_status.json"
    training_env_steps = total_env_steps - warmup_steps

    status = load_json(status_path, {"runs": {}})
    status.update(
        {
            "started_at": status.get("started_at", utc_now()),
            "status": "running",
            "protocol": {
                "total_env_steps": total_env_steps,
                "warmup_steps": warmup_steps,
                "training_env_steps": training_env_steps,
                "gamma": DEFAULT_GAMMA,
                "epsilon": DEFAULT_EPSILON,
                "checkpoint": "final",
            },
        }
    )
    write_json(status_path, status)

    failures = 0
    for layout in args.layouts:
        for seed in args.seeds:
            key = f"{layout}/seed_{seed}"
            try:
                run_dir = run_one(
                    args,
                    layout,
                    seed,
                    exp_name,
                    total_env_steps,
                    warmup_steps,
                    training_env_steps,
                )
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
