#
# Agent: Random - DQN
# Experiment: bandit_noise - all sweeps
# Fail to run in windows env
#
import bsuite
import numpy as np
from bsuite.baselines.dqn import dqn
from bsuite.baselines import experiment
from bsuite import sweep

# make a run of sweep BANDIT_NOISE for agents random and DQN

SAVE_PATH_RAND = './results/rand'
SAVE_PATH_DQN = './results/dqn'


def run_random_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    for episode in range(env.bsuite_num_episodes):
        timestep = env.reset()
        while not timestep.last():
            # take a random action
            action = np.random.choice(env.action_spec().num_values)
            timestep = env.step(action)
    return


def run_dqn_agent(bsuite_id, save_path=SAVE_PATH_DQN, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    agent = dqn.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all sweep
print("Run Random Agents")
for bsuite_id in sweep.BANDIT_NOISE:
    print("sweep: {}".format(bsuite_id))
    run_random_agent(bsuite_id)
print("Run DQN Agents")
for bsuite_id in sweep.BANDIT_NOISE:
    print("sweep: {}".format(bsuite_id))
    run_dqn_agent(bsuite_id)

