# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for RSL-RL configuration compatibility."""

from absl.testing import absltest
from absl.testing import parameterized

from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params


class RslRlConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "locomotion",
          locomotion_params.rsl_rl_config,
          "Go1JoystickFlatTerrain",
      ),
      (
          "manipulation",
          manipulation_params.rsl_rl_config,
          "LeapCubeReorient",
      ),
  )
  def test_uses_rsl_rl_v5_model_schema(self, config_fn, env_name):
    config = config_fn(env_name)

    self.assertNotIn("policy", config)
    self.assertEqual(config.actor.class_name, "MLPModel")
    self.assertEqual(config.actor.hidden_dims, [512, 256, 128])
    self.assertEqual(config.actor.activation, "elu")
    self.assertTrue(config.actor.obs_normalization)
    self.assertEqual(
        config.actor.distribution_cfg.class_name, "GaussianDistribution"
    )
    self.assertEqual(config.actor.distribution_cfg.init_std, 1.0)

    self.assertEqual(config.critic.class_name, "MLPModel")
    self.assertEqual(config.critic.hidden_dims, [512, 256, 128])
    self.assertEqual(config.critic.activation, "elu")
    self.assertTrue(config.critic.obs_normalization)
    self.assertNotIn("distribution_cfg", config.critic)


if __name__ == "__main__":
  absltest.main()
