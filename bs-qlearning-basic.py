#
# Agent: Random
# Experiment: bandit - all sweeps
#
import bsuite
import numpy as np
from bsuite import sweep
from bsuite.baselines import base
from bsuite.baselines import experiment
import dm_env
from dm_env import specs
import numpy as np
from typing import Optional
import math

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
        # minimun learning rate
        self._min_alpha = min_alpha
        # minimun exploration rate
        self._min_epsilon = min_epsilon
        # discount factor
        self._gamma = gamma
        self._rng = np.random.RandomState(seed)
        # initialising Q-table
        self._Q = np.zeros(self._num_actions + (obs_spec.shape,))

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, episode):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((episode + 1) / 25)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, episode):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((episode + 1) / 25)))

    def policy(self, timestep: dm_env.TimeStep) -> base.Action:
        """Select actions according to epsilon-greedy policy."""
        # Epsilon-greedy policy.
        # https: // github.com / deepmind / dm_env / blob / master / docs / index.md
        if self._rng.rand() < self._epsilon:
            return np.random.randint(self._num_actions)
        q_values = self._forward(timestep.observation[None, ...])
        return int(np.argmax(q_values))

    def update(self,
               timestep: dm_env.TimeStep,
               action: base.Action,
               new_timestep: dm_env.TimeStep) -> None:

        self._Q[state_old][action] += alpha * (
                    reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def default_agent(obs_spec: specs.Array,
                      action_spec: specs.DiscreteArray):
        """Initialize a QLearning agent with default parameters."""
        return QLearning(action_spec=action_spec,
                         obs_spec=obs_spec,
                         min_alpha=0.1,
                         min_epsilon=0.1,
                         gamma=1.0,
                         seed=42)


# evaluate a random agent experiment on a single bsuite_id
def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    agent = QLearning.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

