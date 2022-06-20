import supersuit as ss
from pettingzoo.utils import to_parallel
from stable_baselines3 import SAC, A2C, PPO

from envs.commons.commons_multiagent import build_commons_multiagent_env
from envs.commons.commons_regulator import CommonsRegulator
from sd_logging.callbacks import EndOfEpisodeCallback

if __name__ == "__main__":

    env = CommonsRegulator()

    regulator_policy_kwargs = dict(net_arch=[128, 128])
    model = SAC("MlpPolicy", env, gamma=0.99, learning_rate=0.00039, verbose=0, policy_kwargs=regulator_policy_kwargs)

    model.learn(total_timesteps=6000, callback=EndOfEpisodeCallback())