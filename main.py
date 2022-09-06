from stable_baselines3 import A2C, SAC

from envs.utils.env_router import build_env
from sd_logging.callbacks import EndOfEpisodeCallback

if __name__ == "__main__":

    # in case penalty_multiplier is set to a value different from None the penalty_multiplier will remain fixed
    # throughout the simulation. The penalty multiplier var has no effect if active_regulator=False
    env = build_env(active_regulator=True, penalty_multiplier=3)

    regulator_policy_kwargs = dict(net_arch=[128, 128])
    model = SAC("MlpPolicy", env, gamma=0.99, learning_rate=0.00039, verbose=0, policy_kwargs=regulator_policy_kwargs)

    # in case active_regulator=False the model will run for total_timesteps steps
    # in case active_regulator=True the model will run for total_timesteps x regulator_env.agents_steps_per_period steps
    model.learn(total_timesteps=100, callback=EndOfEpisodeCallback())
