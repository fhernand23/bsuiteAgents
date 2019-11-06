#
# Agent: DQN
# Experiment: bandit - all sweeps
#
import bsuite
from bsuite.baselines.dqn_tf2 import dqn
from bsuite.baselines import experiment
from bsuite import sweep

SAVE_PATH_RAND = './bs01/dqntf2'


def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    agent = dqn.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

