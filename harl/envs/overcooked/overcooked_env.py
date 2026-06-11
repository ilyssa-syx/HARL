"""HARL adapter for the Overcooked-AI environment."""

import copy

import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS


REWARD_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


class OvercookedEnv:
    """Expose two-player Overcooked through HARL's multi-agent interface."""

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.layout_name = self.args["layout_name"]
        self.horizon = int(self.args.get("horizon", 400))
        self.n_agents = 2
        self._seed = 0

        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name=self.layout_name,
            rew_shaping_params=REWARD_SHAPING_PARAMS,
        )
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            self.mdp, NO_COUNTERS_PARAMS, force_compute=False
        )
        self.env = BaseOvercookedEnv(self.mdp, horizon=self.horizon)

        initial_obs = self._featurize(self.env.state)
        obs_shape = initial_obs[0].shape
        local_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        shared_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape[0] * self.n_agents,),
            dtype=np.float32,
        )
        self.observation_space = [local_space for _ in range(self.n_agents)]
        self.share_observation_space = [
            shared_space for _ in range(self.n_agents)
        ]
        self.action_space = [
            gym.spaces.Discrete(len(Action.ALL_ACTIONS))
            for _ in range(self.n_agents)
        ]

    def _featurize(self, state):
        return np.asarray(self.mdp.featurize_state(state, self.mlp), dtype=np.float32)

    def _shared_state(self, obs):
        state = np.concatenate(obs, axis=0).astype(np.float32, copy=False)
        return np.repeat(state[np.newaxis, :], self.n_agents, axis=0)

    def get_avail_actions(self):
        return np.ones(
            (self.n_agents, len(Action.ALL_ACTIONS)), dtype=np.float32
        )

    def step(self, actions):
        action_indices = np.asarray(actions).reshape(self.n_agents).astype(int)
        joint_action = tuple(
            Action.INDEX_TO_ACTION[action_index] for action_index in action_indices
        )
        next_state, sparse_reward, done, base_info = self.env.step(joint_action)
        shaped_reward = base_info["shaped_r"]
        total_reward = sparse_reward + shaped_reward

        obs = self._featurize(next_state)
        shared_state = self._shared_state(obs)
        rewards = np.full((self.n_agents, 1), total_reward, dtype=np.float32)
        dones = np.full(self.n_agents, done, dtype=bool)

        info = {
            "sparse_reward": float(sparse_reward),
            "shaped_reward": float(shaped_reward),
            "total_reward": float(total_reward),
        }
        if done:
            sparse_return = float(self.env.cumulative_sparse_rewards)
            shaped_return = float(self.env.cumulative_shaped_rewards)
            info["episode"] = {
                "total_return": sparse_return + shaped_return,
                "sparse_return": sparse_return,
                "shaped_return": shaped_return,
                "deliveries": sparse_return / float(self.mdp.delivery_reward),
                "length": int(self.env.t),
            }

        infos = [copy.deepcopy(info) for _ in range(self.n_agents)]
        return obs, shared_state, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        self.env.reset()
        obs = self._featurize(self.env.state)
        return obs, self._shared_state(obs), self.get_avail_actions()

    def seed(self, seed):
        self._seed = int(seed)
        np.random.seed(self._seed)
        for agent_id, action_space in enumerate(self.action_space):
            action_space.seed(self._seed + agent_id)

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Overcooked adapter only supports text rendering")
        print(self.env)

    def close(self):
        pass
