"""Export one trained HARL HAD3QN Overcooked episode for browser replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from harl.algorithms.actors.had3qn import HAD3QN
from harl.envs.overcooked.overcooked_env import OvercookedEnv


BROWSER_LAYOUT_NAMES = {
    "simple": "cramped_room",
    "unident_s": "asymmetric_advantages",
    "random1": "coordination_ring",
    "random0": "forced_coordination",
    "random3": "counter_circuit",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a HARL HAD3QN episode for PantheonRL's web replay."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Use best_models or the latest/final models checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to run-dir/replay.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_config(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HARL run config: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config["main_args"]["algo"] != "had3qn":
        raise ValueError("Replay export only supports HAD3QN runs")
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
def collect_episode(env, actors):
    obs, _, _ = env.reset()
    states = []
    actions = []
    rewards = []
    dones = []
    done = False
    episode = None

    while not done:
        states.append(env.env.state.to_dict())
        joint_action = []
        for agent_id, actor in enumerate(actors):
            action = actor.get_actions(obs[agent_id][np.newaxis, :], False)
            joint_action.append(int(action.item()))

        obs, _, step_rewards, step_dones, infos, _ = env.step(joint_action)
        done = bool(np.all(step_dones))
        actions.append(joint_action)
        rewards.append(float(step_rewards[0][0]))
        dones.append(done)
        if done:
            episode = infos[0]["episode"]

    return states, actions, rewards, dones, episode


def main():
    args = parse_args()
    config = load_config(args.run_dir)
    env_args = config["env_args"]
    layout_name = env_args["layout_name"]
    if layout_name not in BROWSER_LAYOUT_NAMES:
        raise ValueError(
            f"PantheonRL's browser viewer does not define layout {layout_name!r}"
        )

    output = args.output or (args.run_dir / "replay.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing replay: {output}")

    model_dir = args.run_dir / (
        "best_models" if args.checkpoint == "best" else "models"
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_dir}")

    device = resolve_device(args.device)
    env = OvercookedEnv(env_args)
    env.seed(config["algo_args"]["seed"]["seed"])
    actors = load_actors(config, env, model_dir, device)
    try:
        states, actions, rewards, dones, episode = collect_episode(env, actors)
        mdp_params = env.mdp.mdp_params
    finally:
        env.close()

    browser_mdp_params = {
        "layout_name": BROWSER_LAYOUT_NAMES[layout_name],
        "cook_time": mdp_params["cook_time"],
        "start_order_list": mdp_params["start_order_list"],
        "num_items_for_soup": mdp_params["num_items_for_soup"],
        "delivery_reward": mdp_params["delivery_reward"],
        "rew_shaping_params": mdp_params["rew_shaping_params"],
    }
    replay = {
        "ep_states": [states],
        "ep_observations": [states],
        "ep_actions": [actions],
        "ep_rewards": [rewards],
        "ep_dones": [dones],
        "ep_returns": [episode["total_return"]],
        "ep_returns_sparse": [episode["sparse_return"]],
        "ep_lengths": [episode["length"]],
        "mdp_params": [browser_mdp_params],
        "env_params": [{"horizon": env_args.get("horizon", 400)}],
        "source": {
            "framework": "HARL",
            "algorithm": "had3qn",
            "checkpoint": args.checkpoint,
            "model_dir": str(model_dir),
            "seed": config["algo_args"]["seed"]["seed"],
            "layout_name": layout_name,
            "device": str(device),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(replay, file)
        file.write("\n")
    print(
        f"Wrote {episode['length']}-step replay with "
        f"{episode['deliveries']:.0f} deliveries to {output}"
    )


if __name__ == "__main__":
    main()
