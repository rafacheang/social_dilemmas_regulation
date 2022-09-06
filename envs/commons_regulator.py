import gym
import numpy as np
from gym.spaces import Box
from stable_baselines3 import A2C

from envs.commons_shared import SharedEnv
from envs.utils.utils import normalized_resources, regulator_calculate_sustainability, regulator_normalized_consumption
from sd_logging.logging import Logging


def build_regulator_env(multiagent_env, n_agents, penalty_multiplier):
    return CommonsRegulator(multiagent_env, n_agents, penalty_multiplier)


class CommonsRegulator(gym.Env):

    def __init__(self, multiagent_env, n_agents, penalty_multiplier):

        multiagent_policy_kwargs = dict(net_arch=[64, 64])
        self.multiagent_model = A2C("MlpPolicy", multiagent_env,gamma=0.01, verbose=0,
                                    learning_rate=0.00039,
                                    policy_kwargs=multiagent_policy_kwargs)

        if penalty_multiplier is None:
            # in case fine_multiplier is not explicitly set we let the regulator choose it

            # 1st - limit to exploit increase/decrease, 2nd - penalty multiplier increase/decrease
            self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        else:
            # in case fine_multiplier is explicitly set the regulator is not able to vary it

            # 1st - limit to exploit increase/decrease
            self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 1st is the resources, 2nd is short-term sustainability, 3rd is long-term sustainability
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        # --- # --- # --- # --- # --- # --- # SIMULATION CONSTANTS # --- # --- # --- # --- # --- # --- #

        self.n_agents = n_agents

        self.penalty_multiplier = penalty_multiplier

        # sustainability metrics
        self.n_periods_short_term = 1
        self.n_periods_long_term = 4

        # max periods in one episode
        self.max_periods_per_episode = 10

        # number of steps agents take in between norm set
        self.agents_steps_per_period = 100

        # max consumption, penalty, and increases in consumption and penalty
        self.max_penalty_multiplier_increase = 0.5
        self.max_limit_exploit_increase = 400

        # --- # --- # --- # --- # --- # --- # END OF SIMULATION CONSTANTS # --- # --- # --- # --- # --- # --- #

    def step(self, action):

        SharedEnv.limit_exploit += (action[0] * self.max_limit_exploit_increase)

        # clipping limit_exploit between 0 and max_limit_exploit
        SharedEnv.limit_exploit = max(0, min(SharedEnv.limit_exploit,
                                                SharedEnv.max_consumption))
        # print("new limit was set to: {}".format(SharedEnv.limit_exploit))

        if self.penalty_multiplier is None:
            SharedEnv.penalty_multiplier += (action[1] * self.max_penalty_multiplier_increase)

            # clipping penalty between 0 and max_penalty
            SharedEnv.penalty_multiplier = max(0, min(SharedEnv.penalty_multiplier,
                                                      SharedEnv.max_penalty_multiplier))
            # print("new penalty multiplier was set to: {}".format(SharedEnv.penalty_multiplier))

        # trains participants for self.agents_steps_per_episode
        self.multiagent_model.learn(total_timesteps=self.agents_steps_per_period)

        self.episode_consumption.append(sum(SharedEnv.period_consumption))
        self.episode_replenishment.append(sum(SharedEnv.period_replenishment))

        # calculate sustainability
        st_sustainability = regulator_calculate_sustainability(self.episode_consumption,
                                                               self.episode_replenishment,
                                                               self.n_periods_short_term)
        lt_sustainability = regulator_calculate_sustainability(self.episode_consumption,
                                                               self.episode_replenishment,
                                                               self.n_periods_long_term)

        state = np.array([normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity),
                          st_sustainability, lt_sustainability], dtype=np.float32)

        reward = regulator_normalized_consumption(SharedEnv.max_replenishment,
                                                  self.n_agents,
                                                  self.episode_consumption)  # reward = normalized last period consumption
        # print("regulator's reward: {}".format(reward))

        SharedEnv.periods_counter += 1
        done = bool(SharedEnv.periods_counter > self.max_periods_per_episode
                    or SharedEnv.resources == 0)

        if done:
            Logging.log_episodes_rewards(SharedEnv.episode_number, sum(self.episode_consumption))
            print(f'episodes reward of {Logging.episodes_rewards[-1]} was logged')
            SharedEnv.episode_number += 1

        # should return next_state (observation), reward, done, {}
        return state, reward, done, {}

    def reset(self):
        # print("outer env reset was called")

        SharedEnv.periods_counter = 1

        print(f'episode_number: {SharedEnv.episode_number}')

        SharedEnv.resources = np.random.uniform(low=10000, high=30000)

        st_sustainability = np.random.uniform(low=0.4, high=0.6)
        lt_sustainability = np.random.uniform(low=0.4, high=0.6)

        SharedEnv.limit_exploit = np.random.normal(
            SharedEnv.max_replenishment / (self.n_agents * 2),
            SharedEnv.max_replenishment / (self.n_agents * 8))  # this is ugly

        if self.penalty_multiplier is None:
            SharedEnv.penalty_multiplier = np.random.normal(1, 0.2)
        else:
            SharedEnv.penalty_multiplier = self.penalty_multiplier

        # max consumption per agent
        SharedEnv.max_consumption = SharedEnv.max_replenishment * 2 / self.n_agents  # two times max replenishment per agent

        self.episode_consumption = []
        self.episode_replenishment = []

        state = (normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity),
                 st_sustainability, lt_sustainability)

        return np.array(state, dtype=np.float32)