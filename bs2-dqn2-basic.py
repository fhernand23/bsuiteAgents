#
# Agent: DeepQLearning
# Hyperparameters:
# episodes - a number of games we want the agent to play.
# gamma - aka decay or discount rate, to calculate the future discounted reward.
# epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
# epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
# epsilon_min - we want the agent to explore at least this amount.
# learning_rate - Determines how much neural net learns in each iteration.
#
# Experiment: bandit - all sweeps
#
import bsuite
from bsuite import sweep
from bsuite.baselines import base
from bsuite.baselines import experiment
import dm_env
from dm_env import specs
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from collections import deque
from datetime import datetime


SAVE_PATH_RAND = './bs2/dqn2'


class DQN2(base.Agent):
    """A simple deep qlearning agent."""
    def __init__(self,
                obs_spec: specs.Array,
                action_spec: specs.DiscreteArray,
                model,
                target_model,
                max_memory_length,
                gamma,
                epsilon,
                epsilon_min,
                epsilon_decay,
                learning_rate,
                batch_size,
                tau,
                seed: int = None,):
        # configuration and hyperparameters.
        self._num_actions = action_spec.num_values
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._tau = tau

        self._total_steps = 0
        # self._memory = replay.Replay(capacity=max_memory_length)
        self._memory = deque(maxlen=max_memory_length)
        self._model = model
        self._target_model = target_model

        tf.random.set_seed(seed)
        self._rng = np.random.RandomState(seed)

    def policy(self, timestep: dm_env.TimeStep) -> base.Action:
        # Epsilon-greedy policy.
        if self._total_steps % 50 == 0:
            print("policy: " + str(self._total_steps) + " " + datetime.now().strftime("%H:%M:%S"))
        self._epsilon *= self._epsilon_decay
        self._epsilon = max(self._epsilon_min, self._epsilon)
        if self._rng.rand() < self._epsilon:
            return np.random.randint(self._num_actions)

        # act_values = self._model.predict(timestep.observation[None, ...])
        act_values = self._model.predict(timestep.observation)
        return int(np.argmax(act_values[0]))

    def update(self, timestep: dm_env.TimeStep, action: base.Action, new_timestep: dm_env.TimeStep,):
        # Add this transition to replay.
        if self._total_steps % 50 == 0:
            print("update: " + str(self._total_steps) + " " + datetime.now().strftime("%H:%M:%S"))
        #self._memory.add([timestep.observation, action, new_timestep.reward, new_timestep.discount, new_timestep.observation, new_timestep.last()])
        self._memory.append((timestep.observation, action, new_timestep.reward, new_timestep.discount, new_timestep.observation, new_timestep.last()))

        self._total_steps += 1

        # if self._memory.size > self._batch_size:
        if len(self._memory) > self._batch_size:
            self.replay(self._batch_size)
            self.target_train()  # iterates target model

    def replay(self, batch_size):
        # run a random sample of past actions
        # minibatch = self._memory.sample(batch_size)
        if self._total_steps % 50 == 0:
            print("replay: " + str(self._total_steps) + " " + datetime.now().strftime("%H:%M:%S"))
        minibatch = random.sample(self._memory, batch_size)
        for state, action, reward, discount, next_state, done in minibatch:
            # target = reward
            # if not done:
            #     target = (reward + self._gamma * np.amax(self._model.predict(next_state)[0]))
            # target_f = self._model.predict(state)
            # target_f[0][action] = target
            # self._model.fit(state, target_f, epochs=1, verbose=0)
            target = self._model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self._target_model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self._model.fit(state, target, epochs=1, verbose=0)
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def target_train(self):
        weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self._tau + target_weights[i] * (1 - self._tau)
        self._target_model.set_weights(target_weights)

    def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray):
        """Initialize a DeepQLearning agent with default hyper parameters."""
        learning_rate = 0.005
        # create the model Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=obs_spec.shape[0], activation="relu"))
        model.add(keras.layers.Dense(48, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(action_spec.num_values))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

        target_model = keras.Sequential()
        target_model.add(keras.layers.Dense(24, input_dim=obs_spec.shape[0], activation="relu"))
        target_model.add(keras.layers.Dense(48, activation="relu"))
        target_model.add(keras.layers.Dense(24, activation="relu"))
        target_model.add(keras.layers.Dense(action_spec.num_values))
        target_model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

        return DQN2(obs_spec=obs_spec,
                            action_spec=action_spec,
                            model = model,
                            target_model = target_model,
                            max_memory_length=2000,
                            gamma = 0.85,  # discount rate
                            epsilon = 1.0,  # exploration rate
                            epsilon_min = 0.01,
                            epsilon_decay = 0.995,
                            learning_rate = learning_rate,
                            tau = .125,
                            batch_size = 32,
                            seed=42)


# evaluate the agent experiment on a single bsuite_id
def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}, start={}'.format(
        bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes, datetime.now().strftime("%H:%M:%S")))
    agent = DQN2.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

# run the agents for all CATCH sweeps
for bsuite_id in sweep.CATCH:
    run_agent(bsuite_id)

# run the agents for all MNIST sweeps
for bsuite_id in sweep.MNIST:
    run_agent(bsuite_id)

