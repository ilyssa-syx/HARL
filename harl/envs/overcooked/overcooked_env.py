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

POT_PROGRESS_VALUES = {
    1: 0.3,
    2: 0.6,
    "cooking": 1.0,
    "ready": 1.2,
}
READY_WITH_DISH_BONUS = 0.3
HELD_SOUP_VALUE = 2.0
COUNTER_SOUP_VALUE = 1.8
HELD_SOUP_DISTANCE_COEF = 0.05
COUNTER_SOUP_DISTANCE_COEF = 0.03
DISH_TO_READY_POT_DISTANCE_COEF = 0.03
READY_SOUP_STALE_GRACE = 5
READY_SOUP_STALE_COEF = 0.05
READY_SOUP_STALE_CAP = 0.6
HELD_SOUP_STALE_GRACE = 5
HELD_SOUP_STALE_COEF = 0.03
HELD_SOUP_STALE_CAP = 0.4
COUNTER_STAGING_BONUS = 0.2
COUNTER_STAGING_CAP = 0.6
USELESS_INTERACT_PENALTY = 0.05
USELESS_INTERACT_PENALTY_CAP = 0.5


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
            self.args.get("custom_shaping_scale", 1.2)
        )
        custom_shaping_extra_scale_v2 = self.args.get(
            "custom_shaping_extra_scale_v2",
            self.args.get("custom_shaping_scale_v2", None),
        )
        self.custom_shaping_extra_scale_v2 = (
            self.custom_shaping_scale
            if custom_shaping_extra_scale_v2 is None
            else float(custom_shaping_extra_scale_v2)
        )
        self.custom_shaping_version = int(
            self.args.get("custom_shaping_version", 1)
        )
        if self.custom_shaping_version not in (1, 2):
            raise ValueError("custom_shaping_version must be 1 or 2")
        self.custom_shaping_version_switch_step = self.args.get(
            "custom_shaping_version_switch_step", None
        )
        if self.custom_shaping_version_switch_step is not None:
            self.custom_shaping_version_switch_step = int(
                self.custom_shaping_version_switch_step
            )
            if self.custom_shaping_version_switch_step < 0:
                raise ValueError(
                    "custom_shaping_version_switch_step must be non-negative"
                )
        self.n_agents = 2
        self._seed = 0
        self.cumulative_custom_shaped_rewards = 0.0
        self.custom_shaping_elapsed_steps = 0
        self.ready_soup_ages = {}
        self.held_soup_ages = {}
        self.useless_interact_counts = [0, 0]

        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name=self.layout_name,
            rew_shaping_params=REWARD_SHAPING_PARAMS,
        )
        self.non_edge_counters = self._get_non_edge_counters()
        try:
            self.mlp = MediumLevelPlanner.from_pickle_or_compute(
                self.mdp, NO_COUNTERS_PARAMS, force_compute=False
            )
        except ValueError as error:
            if "unsupported pickle protocol" not in str(error):
                raise
            print("Recomputing planner due to:", error)
            self.mlp = MediumLevelPlanner.from_pickle_or_compute(
                self.mdp, NO_COUNTERS_PARAMS, force_compute=True
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

    def _potential(self, state, distance_fn=None, serving_locations=None):
        if distance_fn is None:
            distance_fn = self._nearest_distance
        if serving_locations is None:
            serving_locations = self.mdp.get_serving_locations()
        dish_available = self._dish_available(state)
        ready_pots = 0
        potential = 0.0

        for pot_pos in self.mdp.get_pot_locations():
            obj = state.objects.get(pot_pos)
            if obj is None or obj.name != "soup" or obj.state is None:
                continue
            _, num_items, cook_time = obj.state
            if num_items < self.mdp.num_items_for_soup:
                potential += POT_PROGRESS_VALUES.get(num_items, 0.0)
            elif cook_time >= self.mdp.soup_cooking_time:
                ready_pots += 1
                potential += POT_PROGRESS_VALUES["ready"]
            else:
                potential += POT_PROGRESS_VALUES["cooking"]

        if ready_pots and dish_available:
            potential += READY_WITH_DISH_BONUS

        ready_pot_positions = self._ready_soup_positions(state)
        for player in state.players:
            if (
                player.held_object is not None
                and player.held_object.name == "soup"
            ):
                potential += HELD_SOUP_VALUE
                potential -= HELD_SOUP_DISTANCE_COEF * distance_fn(
                    player.position, serving_locations
                )
            elif (
                player.held_object is not None
                and player.held_object.name == "dish"
                and ready_pot_positions
            ):
                potential -= DISH_TO_READY_POT_DISTANCE_COEF * distance_fn(
                    player.position, ready_pot_positions
                )

        for pos, obj in state.objects.items():
            if (
                obj.name == "soup"
                and pos not in self.mdp.get_pot_locations()
            ):
                potential += COUNTER_SOUP_VALUE
                potential -= COUNTER_SOUP_DISTANCE_COEF * distance_fn(
                    pos, serving_locations
                )

        return potential

    def _potential_v2(self, state):
        return self._potential(state) + self._potential_v2_extra(state)

    def _potential_v2_extra(self, state):
        if not self._is_late_stage_shaping_state(state):
            return 0.0
        potential = 0.0
        staged_objects = sum(
            1
            for pos, obj in state.objects.items()
            if pos in self.non_edge_counters and obj.name in ("dish", "soup")
        )
        potential += min(
            staged_objects * COUNTER_STAGING_BONUS,
            COUNTER_STAGING_CAP,
        )
        for pot_pos in self._ready_soup_positions(state):
            potential -= self._capped_age_penalty(
                self.ready_soup_ages.get(pot_pos, 0),
                READY_SOUP_STALE_GRACE,
                READY_SOUP_STALE_COEF,
                READY_SOUP_STALE_CAP,
            )
        for player_idx in self._held_soup_player_indices(state):
            potential -= self._capped_age_penalty(
                self.held_soup_ages.get(player_idx, 0),
                HELD_SOUP_STALE_GRACE,
                HELD_SOUP_STALE_COEF,
                HELD_SOUP_STALE_CAP,
            )
        return potential

    def _is_late_stage_shaping_state(self, state):
        if self._ready_soup_positions(state):
            return True
        for player in state.players:
            if (
                player.held_object is not None
                and player.held_object.name == "soup"
            ):
                return True
        return any(
            obj.name == "soup" and pos not in self.mdp.get_pot_locations()
            for pos, obj in state.objects.items()
        )

    def _dish_available(self, state):
        if any(
            player.held_object is not None
            and player.held_object.name == "dish"
            for player in state.players
        ):
            return True
        return any(obj.name == "dish" for obj in state.objects.values())

    def _ready_soup_positions(self, state):
        positions = set()
        for pot_pos in self.mdp.get_pot_locations():
            obj = state.objects.get(pot_pos)
            if obj is None or obj.name != "soup" or obj.state is None:
                continue
            _, num_items, cook_time = obj.state
            if (
                num_items >= self.mdp.num_items_for_soup
                and cook_time >= self.mdp.soup_cooking_time
            ):
                positions.add(pot_pos)
        return positions

    def _held_soup_player_indices(self, state):
        return {
            idx
            for idx, player in enumerate(state.players)
            if (
                player.held_object is not None
                and player.held_object.name == "soup"
            )
        }

    def _next_age_trackers(self, state):
        next_ready_ages = {
            pos: self.ready_soup_ages.get(pos, -1) + 1
            for pos in self._ready_soup_positions(state)
        }
        next_held_ages = {
            idx: self.held_soup_ages.get(idx, -1) + 1
            for idx in self._held_soup_player_indices(state)
        }
        return next_ready_ages, next_held_ages

    def _capped_age_penalty(self, age, grace, coefficient, cap):
        return min(max(age - grace, 0) * coefficient, cap)

    def _nearest_distance(self, position, targets):
        if not targets:
            return 0.0
        x, y = position
        return min(abs(x - tx) + abs(y - ty) for tx, ty in targets)

    def _get_non_edge_counters(self):
        width = len(self.mdp.terrain_mtx[0])
        height = len(self.mdp.terrain_mtx)
        return {
            (x, y)
            for x, y in self.mdp.get_counter_locations()
            if 0 < x < width - 1 and 0 < y < height - 1
        }

    def _object_signature(self, obj):
        if obj is None:
            return None
        if obj.name == "soup" and obj.state is not None:
            soup_type, num_items, _ = obj.state
            return (obj.name, soup_type, num_items)
        return (obj.name, obj.state)

    def _interaction_signature(self, state):
        held_objects = tuple(
            self._object_signature(player.held_object)
            for player in state.players
        )
        interactive_objects = tuple(
            sorted(
                (pos, self._object_signature(obj))
                for pos, obj in state.objects.items()
                if pos in self.mdp.get_counter_locations()
                or pos in self.mdp.get_pot_locations()
            )
        )
        return held_objects, interactive_objects

    def _calculate_useless_interact_penalty(
        self, prev_state, next_state, joint_action, sparse_reward
    ):
        prev_signature = self._interaction_signature(prev_state)
        next_signature = self._interaction_signature(next_state)
        late_stage = (
            self._is_late_stage_shaping_state(prev_state)
            or self._is_late_stage_shaping_state(next_state)
        )
        no_interaction_change = (
            late_stage
            and prev_signature == next_signature
            and float(sparse_reward) == 0.0
        )
        penalty = 0.0
        for idx, action in enumerate(joint_action):
            if action == Action.INTERACT and no_interaction_change:
                self.useless_interact_counts[idx] += 1
                if (
                    self.useless_interact_counts[idx] * USELESS_INTERACT_PENALTY
                    <= USELESS_INTERACT_PENALTY_CAP
                ):
                    penalty -= USELESS_INTERACT_PENALTY
            else:
                self.useless_interact_counts[idx] = 0
        return penalty

    def _active_custom_shaping_version(self):
        if (
            self.custom_shaping_version_switch_step is not None
            and self.custom_shaping_elapsed_steps
            >= self.custom_shaping_version_switch_step
        ):
            return 2
        return self.custom_shaping_version

    def _calculate_custom_shaping(self, prev_state, next_state, done):
        if not self.custom_dense_reward:
            self.ready_soup_ages, self.held_soup_ages = self._next_age_trackers(
                next_state
            )
            return 0.0
        prev_phi = self._potential(prev_state)
        prev_extra_phi = self._potential_v2_extra(prev_state)
        next_ready_ages, next_held_ages = self._next_age_trackers(next_state)
        self.ready_soup_ages = next_ready_ages
        self.held_soup_ages = next_held_ages
        next_phi = self._potential(next_state)
        shaped_reward = self.custom_shaping_scale * (
            self.custom_shaping_gamma * next_phi - prev_phi
        )

        if self._active_custom_shaping_version() == 2:
            next_extra_phi = self._potential_v2_extra(next_state)
            shaped_reward += self.custom_shaping_extra_scale_v2 * (
                self.custom_shaping_gamma * next_extra_phi - prev_extra_phi
            )
        return shaped_reward

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
        active_custom_shaping_version = self._active_custom_shaping_version()
        if self.custom_dense_reward and active_custom_shaping_version == 2:
            custom_shaped_reward += (
                self.custom_shaping_extra_scale_v2
                * self._calculate_useless_interact_penalty(
                    prev_state, next_state, joint_action, sparse_reward
                )
            )
        self.custom_shaping_elapsed_steps += 1
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
        self.useless_interact_counts = [0, 0]
        self.ready_soup_ages, self.held_soup_ages = self._next_age_trackers(
            self.env.state
        )
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
