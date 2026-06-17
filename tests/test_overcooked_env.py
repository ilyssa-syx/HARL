"""Tests for the HARL Overcooked environment adapter."""

import unittest

import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState

from harl.envs.overcooked.overcooked_env import OvercookedEnv
from harl.utils.envs_tools import make_train_env


LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]


class OvercookedEnvTest(unittest.TestCase):
    def make_env(self, layout="simple", horizon=400, custom_dense_reward=False):
        return OvercookedEnv(
            {
                "layout_name": layout,
                "horizon": horizon,
                "state_type": "EP",
                "custom_dense_reward": custom_dense_reward,
                "custom_shaping_gamma": 0.99,
                "custom_shaping_scale": 0.4,
            }
        )

    def state_with(self, env, held=None, counter=None, pot=None):
        state = env.mdp.get_standard_start_state()
        if held is not None:
            held_state = (
                ("onion", env.mdp.num_items_for_soup, env.mdp.soup_cooking_time)
                if held == "soup"
                else None
            )
            state.players[0].set_object(
                ObjectState(held, state.players[0].position, held_state)
            )
        if counter is not None:
            counter_pos = env.mdp.get_counter_locations()[0]
            counter_state = (
                ("onion", env.mdp.num_items_for_soup, env.mdp.soup_cooking_time)
                if counter == "soup"
                else None
            )
            state.add_object(ObjectState(counter, counter_pos, counter_state))
        if pot is not None:
            num_items, cook_time = pot
            pot_pos = env.mdp.get_pot_locations()[0]
            state.add_object(
                ObjectState("soup", pot_pos, ("onion", num_items, cook_time))
            )
        return state

    def test_all_experiment_layouts_reset_with_expected_shapes(self):
        for layout in LAYOUTS:
            with self.subTest(layout=layout):
                env = self.make_env(layout)
                try:
                    obs, shared_state, available_actions = env.reset()
                    self.assertEqual(obs.shape[0], 2)
                    self.assertEqual(shared_state.shape, (2, obs.shape[1] * 2))
                    self.assertEqual(available_actions.shape, (2, 6))
                    np.testing.assert_array_equal(
                        shared_state[0], np.concatenate([obs[0], obs[1]])
                    )
                    np.testing.assert_array_equal(shared_state[0], shared_state[1])
                finally:
                    env.close()

    def test_actions_are_mapped_in_fixed_physical_player_order(self):
        env = self.make_env()
        captured = {}

        def fake_step(joint_action):
            captured["joint_action"] = joint_action
            return env.env.state, 2, False, {"shaped_r": 3}

        env.env.step = fake_step
        try:
            _, _, rewards, _, infos, _ = env.step(np.array([[0], [5]]))
        finally:
            env.close()
        self.assertEqual(
            captured["joint_action"],
            (Action.INDEX_TO_ACTION[0], Action.INDEX_TO_ACTION[5]),
        )
        np.testing.assert_array_equal(rewards, [[5], [5]])
        self.assertEqual(infos[0]["sparse_reward"], 2.0)
        self.assertEqual(infos[0]["built_in_shaped_reward"], 3.0)
        self.assertEqual(infos[0]["custom_shaped_reward"], 0.0)
        self.assertEqual(infos[0]["shaped_reward"], 3.0)
        self.assertEqual(infos[0]["total_reward"], 5.0)

    def test_progress_score_phases(self):
        env = self.make_env()
        try:
            cooking_time = env.mdp.soup_cooking_time
            cases = [
                (self.state_with(env), 0),
                (self.state_with(env, held="onion"), 1),
                (self.state_with(env, pot=(1, 0)), 2),
                (self.state_with(env, pot=(2, 0)), 3),
                (self.state_with(env, pot=(3, 1)), 4),
                (self.state_with(env, held="dish", pot=(3, 1)), 5),
                (self.state_with(env, pot=(3, cooking_time)), 5),
                (
                    self.state_with(
                        env, held="dish", pot=(3, cooking_time)
                    ),
                    6,
                ),
                (self.state_with(env, held="soup"), 7),
            ]
            for state, expected_score in cases:
                with self.subTest(expected_score=expected_score):
                    self.assertEqual(env._get_progress_score(state), expected_score)
        finally:
            env.close()

    def test_custom_shaping_discounts_stalled_progress(self):
        env = self.make_env(custom_dense_reward=True)
        try:
            state = self.state_with(env, pot=(2, 0))
            shaping = env._calculate_custom_shaping(state, state, False)
            self.assertAlmostEqual(shaping, 0.4 * (0.99 * 3.0 - 3.0))
        finally:
            env.close()

    def test_counter_objects_preserve_progress_during_handoffs(self):
        env = self.make_env()
        try:
            cooking_time = env.mdp.soup_cooking_time
            self.assertEqual(
                env._get_progress_score(self.state_with(env, counter="onion")),
                1,
            )
            self.assertEqual(
                env._get_progress_score(
                    self.state_with(env, counter="dish", pot=(3, cooking_time))
                ),
                6,
            )
            self.assertEqual(
                env._get_progress_score(self.state_with(env, counter="soup")),
                7,
            )
        finally:
            env.close()

    def test_custom_shaping_uses_potential_based_reward(self):
        env = self.make_env(custom_dense_reward=True)
        try:
            prev_state = self.state_with(env, held="onion")
            next_state = self.state_with(env, pot=(1, 0))
            prev_phi = 1.0
            next_phi = 2.0
            self.assertAlmostEqual(
                env._calculate_custom_shaping(prev_state, next_state, False),
                0.4 * (0.99 * next_phi - prev_phi),
            )
            self.assertAlmostEqual(
                env._calculate_custom_shaping(prev_state, next_state, True),
                0.4 * (0.99 * next_phi - prev_phi),
            )
            self.assertAlmostEqual(
                env._calculate_custom_shaping(next_state, prev_state, False),
                0.4 * (0.99 * prev_phi - next_phi),
            )
        finally:
            env.close()

    def test_step_and_episode_info_split_built_in_and_custom_shaping(self):
        env = self.make_env(custom_dense_reward=True)
        env.env.state = self.state_with(env, held="onion")
        next_state = self.state_with(env, pot=(1, 0))

        def fake_step(_joint_action):
            env.env.cumulative_sparse_rewards = 20
            env.env.cumulative_shaped_rewards = 3
            return next_state, 20, True, {"shaped_r": 3}

        env.env.step = fake_step
        try:
            _, _, rewards, dones, infos, _ = env.step([4, 4])
            env.reset()
            self.assertEqual(env.cumulative_custom_shaped_rewards, 0.0)
        finally:
            env.close()

        custom_reward = 0.4 * (0.99 * 2.0 - 1.0)
        self.assertTrue(np.all(dones))
        np.testing.assert_allclose(
            rewards, [[23.0 + custom_reward], [23.0 + custom_reward]]
        )
        self.assertAlmostEqual(infos[0]["built_in_shaped_reward"], 3.0)
        self.assertAlmostEqual(infos[0]["custom_shaped_reward"], custom_reward)
        self.assertAlmostEqual(infos[0]["shaped_reward"], 3.0 + custom_reward)
        episode = infos[0]["episode"]
        self.assertAlmostEqual(episode["built_in_shaped_return"], 3.0)
        self.assertAlmostEqual(episode["custom_shaped_return"], custom_reward)
        self.assertAlmostEqual(
            episode["shaped_return"],
            episode["built_in_shaped_return"]
            + episode["custom_shaped_return"],
        )
        self.assertAlmostEqual(
            episode["total_return"],
            episode["sparse_return"] + episode["shaped_return"],
        )

    def test_team_reward_and_terminal_episode_info(self):
        env = self.make_env(horizon=400)
        try:
            for _ in range(399):
                _, _, rewards, dones, infos, _ = env.step([4, 4])
                self.assertFalse(np.any(dones))
                self.assertEqual(float(rewards[0, 0]), float(rewards[1, 0]))
                self.assertNotIn("bad_transition", infos[0])
            _, _, rewards, dones, infos, _ = env.step([4, 4])
        finally:
            env.close()

        self.assertTrue(np.all(dones))
        self.assertEqual(float(rewards[0, 0]), float(rewards[1, 0]))
        self.assertNotIn("bad_transition", infos[0])
        self.assertEqual(infos[0]["episode"]["length"], 400)
        self.assertEqual(
            infos[0]["episode"]["total_return"],
            infos[0]["episode"]["sparse_return"]
            + infos[0]["episode"]["shaped_return"],
        )
        self.assertEqual(
            infos[0]["episode"]["shaped_return"],
            infos[0]["episode"]["built_in_shaped_return"]
            + infos[0]["episode"]["custom_shaped_return"],
        )
        self.assertEqual(infos[0], infos[1])


class OvercookedVectorEnvTest(unittest.TestCase):
    def test_single_and_multi_thread_vector_shapes_and_auto_reset(self):
        args = {"layout_name": "simple", "horizon": 2, "state_type": "EP"}
        for threads in (1, 2):
            with self.subTest(threads=threads):
                envs = make_train_env(
                    "overcooked", seed=0, n_threads=threads, env_args=args
                )
                try:
                    obs, shared_state, available_actions = envs.reset()
                    self.assertEqual(obs.shape[:2], (threads, 2))
                    self.assertEqual(shared_state.shape[:2], (threads, 2))
                    self.assertEqual(available_actions.shape, (threads, 2, 6))
                    actions = np.full((threads, 2, 1), 4)
                    envs.step(actions)
                    obs, shared_state, rewards, dones, infos, available_actions = (
                        envs.step(actions)
                    )
                    self.assertEqual(obs.shape[:2], (threads, 2))
                    self.assertEqual(shared_state.shape[:2], (threads, 2))
                    self.assertEqual(rewards.shape, (threads, 2, 1))
                    self.assertTrue(np.all(dones))
                    self.assertEqual(available_actions.shape, (threads, 2, 6))
                    self.assertIn("original_obs", infos[0][0])
                finally:
                    envs.close()


if __name__ == "__main__":
    unittest.main()
