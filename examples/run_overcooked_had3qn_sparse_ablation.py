"""Run one sparse-reward HAD3QN Overcooked ablation and evaluate it."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train HAD3QN with sparse reward and conservative learning rates. "
            "Training-time epsilon-greedy returns are written to progress.txt."
        )
    )
    parser.add_argument("--layout", default="random0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=500_000)
    parser.add_argument("--warmup-steps", type=int, default=50_000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--checkpoint-interval", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def newest_run(output_dir, layout, exp_name, seed):
    root = output_dir / "overcooked" / layout / "had3qn" / exp_name
    candidates = list(root.glob(f"seed-{seed:05d}-*"))
    if not candidates:
        raise FileNotFoundError(f"Training produced no run directory under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run(command):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    args = parse_args()
    if args.total_env_steps <= 0 or args.eval_episodes <= 0:
        raise ValueError("Step and episode counts must be positive")
    if not 0 <= args.warmup_steps < args.total_env_steps:
        raise ValueError("Warmup steps must be non-negative and below total steps")

    total_env_steps = args.total_env_steps
    warmup_steps = args.warmup_steps
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    horizon = 400
    n_rollout_threads = 20
    train_interval = 50
    eval_interval = 1_000
    if args.smoke_test:
        total_env_steps = min(total_env_steps, 240)
        warmup_steps = min(warmup_steps, 40)
        batch_size = min(batch_size, 32)
        buffer_size = min(buffer_size, 1_000)
        horizon = 40
        n_rollout_threads = 1
        train_interval = 10
        eval_interval = 40

    training_env_steps = total_env_steps - warmup_steps
    exp_name = f"{args.layout}_sparse_reward_low_lr"
    if args.smoke_test:
        exp_name += "_smoke"
    train_command = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "train.py"),
        "--algo",
        "had3qn",
        "--env",
        "overcooked",
        "--exp_name",
        exp_name,
        "--layout_name",
        args.layout,
        "--seed",
        str(args.seed),
        "--num_env_steps",
        str(training_env_steps),
        "--warmup_steps",
        str(warmup_steps),
        "--custom_dense_reward",
        "False",
        "--lr",
        str(args.actor_lr),
        "--critic_lr",
        str(args.critic_lr),
        "--batch_size",
        str(batch_size),
        "--buffer_size",
        str(buffer_size),
        "--exploration_initial_eps",
        "1.0",
        "--exploration_fraction",
        "0.5",
        "--epsilon",
        "0.05",
        "--checkpoint_interval",
        str(args.checkpoint_interval),
        "--use_eval",
        "False",
        "--n_rollout_threads",
        str(n_rollout_threads),
        "--train_interval",
        str(train_interval),
        "--eval_interval",
        str(eval_interval),
        "--horizon",
        str(horizon),
        "--log_dir",
        str(args.output_dir.resolve()),
    ]
    if args.device == "cpu":
        train_command += ["--cuda", "False"]
    elif args.device == "cuda":
        train_command += ["--cuda", "True"]

    run(train_command)
    run_dir = newest_run(args.output_dir, args.layout, exp_name, args.seed)

    eval_command = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "evaluate_overcooked_had3qn.py"),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        "final",
        "--episodes",
        str(args.eval_episodes),
        "--device",
        args.device,
        "--overwrite",
    ]
    run(eval_command)

    metadata = {
        "purpose": "sparse reward and lower learning-rate HAD3QN ablation",
        "run_dir": str(run_dir),
        "training_reward_curve": str(run_dir / "progress.txt"),
        "deterministic_evaluation": str(run_dir / "evaluation.json"),
        "total_env_steps": total_env_steps,
        "warmup_steps": warmup_steps,
        "training_env_steps": training_env_steps,
        "custom_dense_reward": False,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
    }
    with (run_dir / "ablation.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"\nCompleted: {run_dir}")
    print(f"Training epsilon-reward curve: {run_dir / 'progress.txt'}")
    print(f"Final deterministic evaluation: {run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
