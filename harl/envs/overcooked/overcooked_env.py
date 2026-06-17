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
        self.custom_dense_reward = bool(
            self.args.get("custom_dense_reward", False)
        )
        self.custom_shaping_gamma = float(
            self.args.get("custom_shaping_gamma", 0.99)
        )
        self.custom_shaping_scale = float(
            self.args.get("custom_shaping_scale", 0.4)
        )
        self.n_agents = 2
        self._seed = 0
        self.cumulative_custom_shaped_rewards = 0.0

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

    def _get_progress_score(self, state):
        """Return the highest global task-progress phase present in state."""
        pot_states = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(state)
        held_object_names = {
            player.held_object.name
            for player in state.players
            if player.held_object is not None
        }

        loose_ingredient_exists = bool(
            held_object_names.intersection(("onion", "tomato"))
            or counter_objects["onion"]
            or counter_objects["tomato"]
        )
        dish_in_transit = bool(
            "dish" in held_object_names or counter_objects["dish"]
        )
        plated_soup_in_transit = bool(
            "soup" in held_object_names or counter_objects["soup"]
        )

        ready_pots = (
            pot_states["onion"]["ready"] + pot_states["tomato"]["ready"]
        )
        cooking_pots = (
            pot_states["onion"]["cooking"] + pot_states["tomato"]["cooking"]
        )
        one_item_pots = (
            pot_states["onion"]["1_items"] + pot_states["tomato"]["1_items"]
        )
        two_item_pots = (
            pot_states["onion"]["2_items"] + pot_states["tomato"]["2_items"]
        )

        if plated_soup_in_transit:
            return 7
        if ready_pots and dish_in_transit:
            return 6
        if ready_pots or (cooking_pots and dish_in_transit):
            return 5
        if cooking_pots:
            return 4
        if two_item_pots:
            return 3
        if one_item_pots:
            return 2
        if loose_ingredient_exists:
            return 1
        return 0

    def _potential(self, state):
        return self._get_progress_score(state)

    def _calculate_custom_shaping(self, prev_state, next_state, done):
        if not self.custom_dense_reward:
            return 0.0
        prev_phi = self._potential(prev_state)
        next_phi = self._potential(next_state)
        return self.custom_shaping_scale * (
            self.custom_shaping_gamma * next_phi - prev_phi
        )

    def step(self, actions):
        action_indices = np.asarray(actions).reshape(self.n_agents).astype(int)
        joint_action = tuple(
            Action.INDEX_TO_ACTION[action_index] for action_index in action_indices
        )
        prev_state = self.env.state.deepcopy()
        next_state, sparse_reward, done, base_info = self.env.step(joint_action)
        built_in_shaped_reward = float(base_info["shaped_r"])
        custom_shaped_reward = self._calculate_custom_shaping(
            prev_state, next_state, done
        )
        self.cumulative_custom_shaped_rewards += custom_shaped_reward
        shaped_reward = built_in_shaped_reward + custom_shaped_reward
        total_reward = sparse_reward + shaped_reward

        obs = self._featurize(next_state)
        shared_state = self._shared_state(obs)
        rewards = np.full((self.n_agents, 1), total_reward, dtype=np.float32)
        dones = np.full(self.n_agents, done, dtype=bool)

        info = {
            "sparse_reward": float(sparse_reward),
            "built_in_shaped_reward": built_in_shaped_reward,
            "custom_shaped_reward": float(custom_shaped_reward),
            "shaped_reward": float(shaped_reward),
            "total_reward": float(total_reward),
        }
        if done:
            sparse_return = float(self.env.cumulative_sparse_rewards)
            built_in_shaped_return = float(self.env.cumulative_shaped_rewards)
            custom_shaped_return = float(self.cumulative_custom_shaped_rewards)
            shaped_return = built_in_shaped_return + custom_shaped_return
            info["episode"] = {
                "total_return": sparse_return + shaped_return,
                "sparse_return": sparse_return,
                "built_in_shaped_return": built_in_shaped_return,
                "custom_shaped_return": custom_shaped_return,
                "shaped_return": shaped_return,
                "deliveries": sparse_return / float(self.mdp.delivery_reward),
                "length": int(self.env.t),
            }

        infos = [copy.deepcopy(info) for _ in range(self.n_agents)]
        return obs, shared_state, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        self.env.reset()
        self.cumulative_custom_shaped_rewards = 0.0
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
