#
# Agent: Random
# Experiment: bandit - all sweeps
#
import bsuite
import numpy as np
from bsuite import sweep
from bsuite.baselines import base
import dm_env
from dm_env import specs
import numpy as np
from typing import Optional

SAVE_PATH_RAND = './bs01/ql'


class QLearning(base.Agent):
  """A qlearning agent."""

  def __init__(self,
               obs_spec: specs.Array,
               action_spec: specs.DiscreteArray,
               min_alpha,
               min_epsilon,
               gamma,
               seed: int = None,):
    self._num_actions = action_spec.num_values
    self._rng = np.random.RandomState(seed)

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    del timestep
    return self._rng.randint(self._num_actions)

  def update(self,
             timestep: dm_env.TimeStep,
             action: base.Action,
             new_timestep: dm_env.TimeStep) -> None:
    del timestep
    del action
    del new_timestep


def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray,
                  **kwargs) -> Random:
  del obs_spec  # for compatibility
  return Random(action_spec=action_spec, **kwargs)

# evaluate a random agent experiment on a single bsuite_id
def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    for episode in range(env.bsuite_num_episodes):
        timestep = env.reset()
        while not timestep.last():
            action = np.random.choice(env.action_spec().num_values)
            timestep = env.step(action)
    return


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

