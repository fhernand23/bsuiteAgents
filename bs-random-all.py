#
# Agent: Random
# Experiment: All
#
import bsuite
from bsuite.baselines.random import random
from bsuite.baselines import experiment
from bsuite import sweep

SAVE_PATH_RAND = './bs/rand'

# evaluate a random agent experiment on a single bsuite_id
def run_random_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    agent = random.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# Basic
for bsuite_id in sweep.BANDIT:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MNIST:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CATCH:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MOUNTAIN_CAR:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CARTPOLE:
    run_random_agent(bsuite_id)

# Reward noise
for bsuite_id in sweep.BANDIT_NOISE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MNIST_NOISE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CATCH_NOISE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MOUNTAIN_CAR_NOISE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CARTPOLE_NOISE:
    run_random_agent(bsuite_id)

# Reward scale
for bsuite_id in sweep.BANDIT_SCALE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MNIST_SCALE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CATCH_SCALE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MOUNTAIN_CAR_SCALE:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CARTPOLE_SCALE:
    run_random_agent(bsuite_id)

# Exploration
for bsuite_id in sweep.DEEP_SEA:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.DEEP_SEA_STOCHASTIC:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.CARTPOLE_SWINGUP:
    run_random_agent(bsuite_id)

# Credit assignment
for bsuite_id in sweep.UMBRELLA_LENGTH:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.UMBRELLA_DISTRACT:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.DISCOUNTING_CHAIN:
    run_random_agent(bsuite_id)

# Memory
for bsuite_id in sweep.MEMORY_LEN:
    run_random_agent(bsuite_id)
for bsuite_id in sweep.MEMORY_SIZE:
    run_random_agent(bsuite_id)

