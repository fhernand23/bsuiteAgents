#
# Agent: Actor Critic RNN
# Experiment: bandit - all sweeps
#
import bsuite
from bsuite.baselines.actor_critic_rnn import actor_critic_rnn
from bsuite.baselines import experiment
from bsuite import sweep

SAVE_PATH_RAND = './bs01/a2crnn'


def run_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    # Load environment
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    agent = actor_critic_rnn.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)


# run the agents for all BANDIT sweeps
for bsuite_id in sweep.BANDIT:
    run_agent(bsuite_id)

