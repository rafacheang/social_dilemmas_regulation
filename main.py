from stable_baselines3 import SAC

from envs.commons.commons_regulator import CommonsRegulator
from envs.commons.commons_shared import CommonsShared

if __name__ == "__main__":

    env = CommonsRegulator()

    model = SAC("MlpPolicy", env, gamma=1, verbose=0)
    model.learn(total_timesteps=100)