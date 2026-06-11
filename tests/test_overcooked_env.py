"""Tests for the HARL Overcooked environment adapter."""

import unittest

import numpy as np
from overcooked_ai_py.mdp.actions import Action

from harl.envs.overcooked.overcooked_env import OvercookedEnv
from harl.utils.envs_tools import make_train_env


LAYOUTS = ["simple", "unident_s", "random1", "random0", "random3"]


class OvercookedEnvTest(unittest.TestCase):
    def make_env(self, layout="simple", horizon=400):
        return OvercookedEnv(
            {"layout_name": layout, "horizon": horizon, "state_type": "EP"}
        )

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
        self.assertEqual(infos[0]["shaped_reward"], 3.0)
        self.assertEqual(infos[0]["total_reward"], 5.0)

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
