#
# Agent: Random
# Experiment: bandit - all sweeps
#
import bsuite
from bsuite.baselines.random import random
from bsuite.baselines import experiment
from bsuite import sweep
from bsuite.baselines import base
import dm_env
from dm_env import specs
import numpy as np
from typing import Optional

SAVE_PATH_RAND = './bs01/rand'


class RandomAgent(base.Agent):
    """A random agent."""

    def __init__(self,
                 obs_spec: specs.Array,
                 action_spec: specs.DiscreteArray,
                 seed: Optional[int] = None):
        self._obs_spec_shape = obs_spec.shape # (1, 1)
        self._num_actions = action_spec.num_values # 11
        self._rng = np.random.RandomState(seed)
        # print("init num_actions: " + str(self._num_actions))
        # print("init obs shape: " + str(obs_spec.shape))

    def policy(self, timestep: dm_env.TimeStep) -> base.Action:
        # print("Policy --------")
        # print("Policy timestep: " + str(timestep)) # TimeStep(step_type=<StepType.FIRST: 0>, reward=None, discount=None, observation=array([[1.]], dtype=float32))
        del timestep
        return self._rng.randint(self._num_actions)
        # print("Policy --------")

    def update(self,
               timestep: dm_env.TimeStep,
               action: base.Action,
               new_timestep: dm_env.TimeStep) -> None:
        # print("Update --------")
        # print("Update timestep: " + str(timestep)) # TimeStep(step_type=<StepType.FIRST: 0>, reward=None, discount=None, observation=array([[1.]], dtype=float32))
        # print("Update action: " + str(action)) # 7
        # print("Update new_timestep: " + str(new_timestep)) # TimeStep(step_type=<StepType.LAST: 2>, reward=0.8, discount=0.0, observation=array([[1.]], dtype=float32))
        # print("Update timestep.observation: " + str(timestep.observation[None, ...]))
        del timestep
        del action
        del new_timestep
        # print("Policy --------")

    def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray):
        """Initialize a QLearning agent with default parameters."""
        return RandomAgent(action_spec=action_spec,
                         obs_spec=obs_spec,
                         seed=42)


# evaluate a random agent experiment on a single bsuite_id
def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    agent = RandomAgent.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
# for bsuite_id in sweep.BANDIT:
#     run_agent(bsuite_id)

# run the agents for all MNIST sweeps
for bsuite_id in sweep.MNIST:
    run_agent(bsuite_id)

# run the agents for all CATCH sweeps
for bsuite_id in sweep.CATCH:
    run_agent(bsuite_id)


