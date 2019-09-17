import bsuite

from bsuite import sweep

# Instantiate the environment corresponding to a given `bsuite_id`
env = bsuite.load_from_id("bandit_noise/0")

# Default configuration is dm_env
import dm_env
env = bsuite.load_from_id(bsuite_id='bandit_noise/0')
isinstance(env, dm_env.Environment)

