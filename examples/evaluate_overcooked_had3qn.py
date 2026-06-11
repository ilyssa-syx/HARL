"""Deterministically evaluate trained HARL HAD3QN actors on Overcooked."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
import torch

from harl.algorithms.actors.had3qn import HAD3QN
from harl.envs.overcooked.overcooked_env import OvercookedEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a completed HARL HAD3QN Overcooked run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Evaluate best_models or the latest/final models checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def summarize(values):
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def load_config(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HARL run config: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config["main_args"]["algo"] != "had3qn":
        raise ValueError("Evaluation only supports HAD3QN runs")
    if config["main_args"]["env"] != "overcooked":
        raise ValueError("Run config is not an Overcooked experiment")
    return config


def load_actors(config, env, model_dir, device):
    actor_args = {**config["algo_args"]["model"], **config["algo_args"]["algo"]}
    actors = []
    for agent_id in range(env.n_agents):
        actor = HAD3QN(
            actor_args,
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )
        actor.restore(model_dir, agent_id)
        actors.append(actor)
    return actors


@torch.no_grad()
def run_episode(env, actors):
    obs, _, _ = env.reset()
    done = False
    episode = None
    while not done:
        actions = []
        for agent_id, actor in enumerate(actors):
            action = actor.get_actions(obs[agent_id][np.newaxis, :], False)
            actions.append(int(action.item()))
        obs, _, _, dones, infos, _ = env.step(actions)
        done = bool(np.all(dones))
        if done:
            episode = infos[0]["episode"]
    return episode


def write_results(output, results):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "total_return",
                "sparse_return",
                "shaped_return",
                "deliveries",
            ]
        )
        for episode_index, episode in enumerate(results["episode_results"]):
            writer.writerow(
                [
                    episode_index,
                    episode["total_return"],
                    episode["sparse_return"],
                    episode["shaped_return"],
                    episode["deliveries"],
                ]
            )


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    output = args.output or (args.run_dir / "evaluation.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")

    config = load_config(args.run_dir)
    env_args = config["env_args"]
    device = resolve_device(args.device)
    model_dir = args.run_dir / (
        "best_models" if args.checkpoint == "best" else "models"
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"{args.checkpoint} checkpoint directory does not exist: {model_dir}"
        )
    env = OvercookedEnv(env_args)
    env.seed(config["algo_args"]["seed"]["seed"])
    actors = load_actors(config, env, model_dir, device)
    try:
        episodes = [run_episode(env, actors) for _ in range(args.episodes)]
    finally:
        env.close()

    metric_keys = ("total_return", "sparse_return", "shaped_return", "deliveries")
    metric_values = {
        metric: [episode[metric] for episode in episodes] for metric in metric_keys
    }
    summary = {
        metric: summarize(values) for metric, values in metric_values.items()
    }
    results = {
        "algo": "had3qn",
        "source": "harl",
        "layout": env_args["layout_name"],
        "seed": config["algo_args"]["seed"]["seed"],
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "model_dir": str(model_dir),
        "episodes": args.episodes,
        "deterministic": True,
        "device": str(device),
        "training_num_env_steps": config["algo_args"]["train"]["num_env_steps"],
        "episode_results": episodes,
        "episode_returns": metric_values["total_return"],
        "episode_sparse_returns": metric_values["sparse_return"],
        "episode_shaped_returns": metric_values["shaped_return"],
        "episode_deliveries": metric_values["deliveries"],
        "summary": summary,
        "mean_return": summary["total_return"]["mean"],
        "std_return": summary["total_return"]["std"],
        "median_return": summary["total_return"]["median"],
        "min_return": summary["total_return"]["min"],
        "max_return": summary["total_return"]["max"],
        "mean_sparse_return": summary["sparse_return"]["mean"],
        "mean_shaped_return": summary["shaped_return"]["mean"],
        "mean_deliveries": summary["deliveries"]["mean"],
    }
    write_results(output, results)
    print(f"Wrote deterministic evaluation to {output}")


if __name__ == "__main__":
    main()
