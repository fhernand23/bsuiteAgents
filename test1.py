import bsuite

from bsuite import sweep

# Valid Ids across all experiments:
print('All possible values for bsuite_id:')
print(sweep.SWEEP)

# Ids for an example experiment:
print('List bsuite_id for "bandit_noise" experiment:')
print(sweep.BANDIT_NOISE)

# List the configurations for the given experiment
for bsuite_id in sweep.BANDIT_NOISE:
  env = bsuite.load_from_id(bsuite_id)
  print('bsuite_id={}, settings={}, num_episodes={}'
        .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))