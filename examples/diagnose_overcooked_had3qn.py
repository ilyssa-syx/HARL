"""Diagnose random and deterministic HAD3QN behavior on Overcooked."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from overcooked_ai_py.mdp.actions import Action

from harl.algorithms.actors.had3qn import HAD3QN
from harl.envs.overcooked.overcooked_env import OvercookedEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["step_0070000", "step_0250000", "step_0490000", "final"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_config(run_dir):
    with (run_dir / "config.json").open(encoding="utf-8") as file:
        return json.load(file)


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


def checkpoint_dir(run_dir, checkpoint):
    if checkpoint == "final":
        return run_dir / "models"
    if checkpoint == "best":
        return run_dir / "best_models"
    return run_dir / "checkpoints" / checkpoint


def state_signature(state):
    players = tuple(
        (
            tuple(player.position),
            tuple(player.orientation),
            None if player.held_object is None else player.held_object.name,
        )
        for player in state.players
    )
    objects = tuple(
        sorted(
            (
                tuple(position),
                obj.name,
                repr(getattr(obj, "state", None)),
            )
            for position, obj in state.objects.items()
        )
    )
    return players, objects


def summarize_counter(counter):
    total = sum(counter.values())
    return {
        str(key): {"count": count, "fraction": count / total}
        for key, count in counter.most_common()
    }


@torch.no_grad()
def run_condition(config, condition, actors, episodes, seed):
    env = OvercookedEnv(config["env_args"])
    env.seed(seed)
    action_names = [str(action) for action in Action.ALL_ACTIONS]
    joint_actions = Counter()
    agent_actions = [Counter() for _ in range(env.n_agents)]
    step_reward_counts = Counter()
    returns = []
    progress_maxima = []
    unchanged_steps = 0
    total_steps = 0
    repeat_runs = []
    q_margins = [[] for _ in range(env.n_agents)]

    try:
        for _ in range(episodes):
            obs, _, _ = env.reset()
            done = False
            last_joint_action = None
            repeat_run = 0
            max_progress = env._get_progress_score(env.env.state)

            while not done:
                before = state_signature(env.env.state)
                if condition == "random":
                    actions = [
                        int(env.action_space[agent_id].sample())
                        for agent_id in range(env.n_agents)
                    ]
                else:
                    actions = []
                    for agent_id, actor in enumerate(actors):
                        obs_tensor = torch.as_tensor(
                            obs[agent_id][np.newaxis, :],
                            dtype=torch.float32,
                            device=actor.tpdv["device"],
                        )
                        q_values = actor.actor(obs_tensor)
                        top_two = torch.topk(q_values, k=2, dim=-1).values
                        q_margins[agent_id].append(
                            float((top_two[0, 0] - top_two[0, 1]).item())
                        )
                        actions.append(int(q_values.argmax(dim=-1).item()))

                joint = tuple(actions)
                joint_actions[joint] += 1
                for agent_id, action in enumerate(actions):
                    agent_actions[agent_id][action_names[action]] += 1

                if joint == last_joint_action:
                    repeat_run += 1
                else:
                    if repeat_run:
                        repeat_runs.append(repeat_run)
                    repeat_run = 1
                    last_joint_action = joint

                obs, _, _, dones, infos, _ = env.step(actions)
                total_steps += 1
                unchanged_steps += state_signature(env.env.state) == before
                max_progress = max(max_progress, env._get_progress_score(env.env.state))
                info = infos[0]
                for key in (
                    "sparse_reward",
                    "built_in_shaped_reward",
                    "custom_shaped_reward",
                    "total_reward",
                ):
                    value = info[key]
                    if value > 0:
                        step_reward_counts[f"{key}:positive"] += 1
                    elif value < 0:
                        step_reward_counts[f"{key}:negative"] += 1
                done = bool(np.all(dones))
                if done:
                    returns.append(info["episode"])
                    progress_maxima.append(max_progress)
                    repeat_runs.append(repeat_run)
    finally:
        env.close()

    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "mean_total_return": float(np.mean([x["total_return"] for x in returns])),
        "mean_sparse_return": float(np.mean([x["sparse_return"] for x in returns])),
        "mean_built_in_shaped_return": float(
            np.mean([x["built_in_shaped_return"] for x in returns])
        ),
        "mean_custom_shaped_return": float(
            np.mean([x["custom_shaped_return"] for x in returns])
        ),
        "max_total_return": float(max(x["total_return"] for x in returns)),
        "max_sparse_return": float(max(x["sparse_return"] for x in returns)),
        "max_progress_score": int(max(progress_maxima)),
        "progress_score_counts": summarize_counter(Counter(progress_maxima)),
        "unchanged_state_fraction": unchanged_steps / total_steps,
        "longest_repeated_joint_action": max(repeat_runs),
        "step_reward_counts": dict(step_reward_counts),
        "joint_actions": summarize_counter(joint_actions),
        "agent_actions": [summarize_counter(counter) for counter in agent_actions],
        "mean_q_margin": [
            None if not margins else float(np.mean(margins)) for margins in q_margins
        ],
        "episode_returns": returns,
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    output = args.output or args.run_dir / "diagnostics.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")

    config = load_config(args.run_dir)
    device = resolve_device(args.device)
    results = {
        "run_dir": str(args.run_dir),
        "episodes_per_condition": args.episodes,
        "device": str(device),
        "conditions": {},
    }
    results["conditions"]["random"] = run_condition(
        config, "random", None, args.episodes, seed=10000
    )

    for checkpoint in args.checkpoints:
        model_dir = checkpoint_dir(args.run_dir, checkpoint)
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        probe_env = OvercookedEnv(config["env_args"])
        actors = load_actors(config, probe_env, model_dir, device)
        probe_env.close()
        results["conditions"][checkpoint] = run_condition(
            config, checkpoint, actors, args.episodes, seed=10000
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Wrote diagnostics to {output}")


if __name__ == "__main__":
    main()
