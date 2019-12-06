#
# Agent: Actor Critic RNN
# Experiment: bandit - all sweeps
#
import bsuite
from bsuite.baselines.actor_critic_rnn import actor_critic_rnn
from bsuite.baselines import experiment
from bsuite import sweep
from datetime import datetime


SAVE_PATH_RAND = './bs01/a2crnn'


def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}, start={}'.format(
        bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes, datetime.now().strftime("%H:%M:%S")))
    agent = actor_critic_rnn.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all MNIST sweeps
for bsuite_id in sweep.MNIST:
    run_agent(bsuite_id)

