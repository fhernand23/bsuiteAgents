import bsuite

from bsuite import sweep
import gym

from bsuite.utils import gym_wrapper
raw_env = bsuite.load_from_id(bsuite_id='memory_len/0')
env = gym_wrapper.GymWrapper(raw_env)
isinstance(env, gym.Env)


