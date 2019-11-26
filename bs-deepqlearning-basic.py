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
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
from bsuite.baselines.utils import replay
import tensorflow.keras.models
import random
from collections import deque

SAVE_PATH_RAND = './bs01/dql'


class DeepQLearning(base.Agent):
    """A simple deep qlearning agent."""
    def __init__(self,
                obs_spec: specs.Array,
                action_spec: specs.DiscreteArray,
                model,
                max_memory_length,
                gamma,
                epsilon,
                epsilon_min,
                epsilon_decay,
                learning_rate,
                batch_size,
                seed: int = None,):
        # configuration and hyperparameters.
        self._num_actions = action_spec.num_values
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        self._total_steps = 0
        # self._memory = replay.Replay(capacity=max_memory_length)
        self._memory = deque(maxlen=max_memory_length)
        self._model = model

        tf.random.set_seed(seed)
        self._rng = np.random.RandomState(seed)

    def policy(self, timestep: dm_env.TimeStep) -> base.Action:
        # Epsilon-greedy policy.
        print("policy: " + str(self._total_steps))
        if self._rng.rand() < self._epsilon:
            return np.random.randint(self._num_actions)
        # print("space tuple: " + str(tuple(timestep.observation[None, ...])))
        # print("observation: " + str(timestep.observation[None, ...]))
        # print("observation type: " + str(type(timestep.observation[None, ...])))
        act_values = self._model.predict(timestep.observation[None, ...])
        return int(np.argmax(act_values[0]))

    def update(self, timestep: dm_env.TimeStep, action: base.Action, new_timestep: dm_env.TimeStep,):
        # Add this transition to replay.
        #self._memory.add([timestep.observation, action, new_timestep.reward, new_timestep.discount, new_timestep.observation, new_timestep.last()])
        self._memory.append((timestep.observation, action, new_timestep.reward, new_timestep.discount, new_timestep.observation, new_timestep.last()))

        self._total_steps += 1

        print("update: " + str(self._total_steps))
        print("last?: " + str(new_timestep.last()))
        # if done:
        #     print("episode: {}/{}, score: {}, e: {:.2}"
        #           .format(e, EPISODES, time, agent.epsilon))
        #     break

        # if self._memory.size > self._batch_size:
        if len(self._memory) > self._batch_size:
            self.replay(self._batch_size)

    def replay(self, batch_size):
        # run a random sample of past actions
        # minibatch = self._memory.sample(batch_size)
        minibatch = random.sample(self._memory, batch_size)
        # print("minibatch type: " + str(type(minibatch)))
        # print("minibatch: " + str(minibatch))
        # TODO cambiar estos valores por los correctos
        for state, action, reward, discount, next_state, done in minibatch:
            print("state: " + str(state))
            print("action: " + str(action))
            print("reward: " + str(reward))
            print("discount: " + str(discount))
            print("next_state: " + str(next_state))
            print("done: " + str(done))
            target = reward
            if not done:
                target = (reward + self._gamma * np.amax(self._model.predict(next_state)[0]))
            target_f = self._model.predict(state)
            target_f[0][action] = target
            self._model.fit(state, target_f, epochs=1, verbose=0)
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray, learning_rate = 0.001):
        """Initialize a DeepQLearning agent with default hyper parameters."""
        # create the model Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        #model.add(keras.layers.Dense(24, input_dim=obs_spec.shape, activation='relu'))
        print("input shape: " + str(obs_spec.shape))
        model.add(keras.layers.Flatten(input_shape=obs_spec.shape))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(action_spec.num_values, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

        return DeepQLearning(obs_spec=obs_spec,
                            action_spec=action_spec,
                            model = model,
                            max_memory_length=2000,
                            gamma = 0.95,  # discount rate
                            epsilon = 1.0,  # exploration rate
                            epsilon_min = 0.01,
                            epsilon_decay = 0.995,
                            learning_rate = learning_rate,
                            batch_size = 32,
                            seed=42)


# evaluate the agent experiment on a single bsuite_id
def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    agent = DeepQLearning.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

# run the agents for all MNIST sweeps
# for bsuite_id in sweep.MNIST:
#     run_agent(bsuite_id)

# run the agents for all CATCH sweeps
# for bsuite_id in sweep.CATCH:
#     run_agent(bsuite_id)

# run the agents for all MOUNTAIN_CAR sweeps
# for bsuite_id in sweep.MOUNTAIN_CAR:
#     run_agent(bsuite_id)
