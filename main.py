import supersuit as ss
from pettingzoo.utils import to_parallel
from stable_baselines3 import SAC, A2C

from envs.commons.commons_multiagent import build_commons_multiagent_env
from envs.commons.commons_regulator import CommonsRegulator
from sd_logging.rewards_callback import RewardsCallback

if __name__ == "__main__":

    rewards_callback = RewardsCallback()

    env = CommonsRegulator()

    model = SAC("MlpPolicy", env, gamma=0.95, verbose=0)
    model.learn(total_timesteps=30000, callback=rewards_callback)