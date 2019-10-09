#
# Agent: Random
# Experiment: bandit_noise - all sweeps
#
import bsuite
import numpy as np

from bsuite import sweep

SAVE_PATH_RAND = './bs01/rand'

# evaluate a random agent experiment on a single bsuite_id
def run_random_agent(bsuite_id, save_path=SAVE_PATH_RAND, overwrite=True):
    env = bsuite.load_and_record(bsuite_id, save_path, overwrite=overwrite)
    print('bsuite_id={}, settings={}, num_episodes={}'.format(bsuite_id, sweep.SETTINGS[bsuite_id],
                                                              env.bsuite_num_episodes))
    for episode in range(env.bsuite_num_episodes):
        timestep = env.reset()
        while not timestep.last():
            action = np.random.choice(env.action_spec().num_values)
            timestep = env.step(action)
    return

for bsuite_id in sweep.BANDIT_NOISE:
    run_random_agent(bsuite_id)

# We have logged results as CSV files in SAVE_PATH
print("Loading results...")
from bsuite.logging import csv_load
DF, _ = csv_load.load_bsuite(SAVE_PATH_RAND)

# We can make use of bsuite summary scoring... and the random agent scores essentially zero
print("Score...")
from bsuite.experiments import summary_analysis
BSUITE_SCORE = summary_analysis.bsuite_score(DF)

# As well as plots specialized to the experiment
print("Plot...")
bandit_noise_df = DF[DF.bsuite_env == 'bandit_noise'].copy()
summary_analysis.plot_single_experiment(BSUITE_SCORE, 'bandit_noise')

# average regret over learning (lower is better)
print("Plot average...")
from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
bandit_noise_analysis.plot_average(bandit_noise_df)
