#
# Agent: DQN
# Experiment: bandit - all sweeps
#
import bsuite
from bsuite.baselines import experiment
from bsuite import sweep
from bsuite.baselines import base
from bsuite.baselines.utils import replay
import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow.compat.v2 as tf


SAVE_PATH_RAND = './bs01/dqntf2'


class DQNTF2(base.Agent):
    """A simple DQN agent using TF2."""

    def __init__(
            self,
            action_spec: specs.DiscreteArray,
            online_network: snt.Module,
            target_network: snt.Module,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            optimizer: snt.Optimizer,
            epsilon: float,
            seed: int = None,):
        # DQN configuration and hyperparameters.
        self._num_actions = action_spec.num_values
        self._discount = discount
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._optimizer = optimizer
        self._epsilon = epsilon
        self._total_steps = 0
        self._replay = replay.Replay(capacity=replay_capacity)
        self._min_replay_size = min_replay_size

        tf.random.set_seed(seed)
        self._rng = np.random.RandomState(seed)

        # Internalize the networks.
        self._online_network = online_network
        self._target_network = target_network
        self._forward = tf.function(online_network)

    def policy(self, timestep: dm_env.TimeStep) -> base.Action:
        # Epsilon-greedy policy.
        if self._rng.rand() < self._epsilon:
            return np.random.randint(self._num_actions)
        q_values = self._forward(timestep.observation[None, ...])
        return int(np.argmax(q_values))

    def update(
            self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            new_timestep: dm_env.TimeStep,):
        # Add this transition to replay.
        self._replay.add([
            timestep.observation,
            action,
            new_timestep.reward,
            new_timestep.discount,
            new_timestep.observation,
        ])

        self._total_steps += 1
        if self._total_steps % self._sgd_period != 0:
            return

        if self._replay.size < self._min_replay_size:
            return

        # Do a batch of SGD.
        transitions = self._replay.sample(self._batch_size)
        self._training_step(transitions)

        # Periodically update target network variables.
        if self._total_steps % self._target_update_period == 0:
            for target, param in zip(self._target_network.trainable_variables,
                                     self._online_network.trainable_variables):
                target.assign(param)

    @tf.function
    def _training_step(self, transitions):
        with tf.GradientTape() as tape:
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            r_t = tf.cast(r_t, tf.float32)  # [B]
            d_t = tf.cast(d_t, tf.float32)  # [B]
            q_tm1 = self._online_network(o_tm1)  # [B, A]
            q_t = self._target_network(o_t)  # [B, A]

            onehot_actions = tf.one_hot(a_tm1, depth=self._num_actions)  # [B, A]
            qa_tm1 = tf.reduce_sum(q_tm1 * onehot_actions, axis=-1)  # [B]
            qa_t = tf.reduce_max(q_t, axis=-1)  # [B]

            # One-step Q-learning loss.
            target = r_t + d_t * self._discount * qa_t
            td_error = qa_tm1 - target
            loss = 0.5 * tf.reduce_sum(td_error ** 2)  # []

        params = self._online_network.trainable_variables
        grads = tape.gradient(loss, params)
        self._optimizer.apply(grads, params)
        return loss

    def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray):
        """Initialize a DQN agent with default parameters."""
        del obs_spec  # Unused.
        hidden_units = [50, 50]
        online_network = snt.Sequential([
            snt.Flatten(),
            snt.nets.MLP(hidden_units + [action_spec.num_values]),
        ])
        target_network = snt.Sequential([
            snt.Flatten(),
            snt.nets.MLP(hidden_units + [action_spec.num_values]),
        ])
        return DQNTF2(
            action_spec=action_spec,
            online_network=online_network,
            target_network=target_network,
            batch_size=32,
            discount=0.99,
            replay_capacity=10000,
            min_replay_size=100,
            sgd_period=1,
            target_update_period=4,
            optimizer=snt.optimizers.Adam(learning_rate=1e-3),
            epsilon=0.05,
            seed=42)


# evaluate the agent experiment on a single bsuite_id
def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    agent = DQNTF2.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

# run the agents for all MNIST sweeps
for bsuite_id in sweep.MNIST:
    run_agent(bsuite_id)

# run the agents for all CATCH sweeps
for bsuite_id in sweep.CATCH:
    run_agent(bsuite_id)
