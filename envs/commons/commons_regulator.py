import gym
import numpy as np
import supersuit as ss
from gym.spaces import Box
from pettingzoo.utils import to_parallel
from stable_baselines3 import A2C, SAC, PPO

from envs.commons.commons_multiagent import build_commons_multiagent_env
from envs.commons.commons_shared import CommonsShared
from sd_logging.logging import Logging


class CommonsRegulator(gym.Env):

    def __init__(self):

        # multiagent environment
        self.multiagent_env = build_commons_multiagent_env()
        self.multiagent_env = to_parallel(self.multiagent_env)
        self.multiagent_env = ss.pettingzoo_env_to_vec_env_v1(self.multiagent_env)
        self.multiagent_env = ss.concat_vec_envs_v1(self.multiagent_env, 1, base_class='stable_baselines3')

        # multiagent model
        multiagent_policy_kwargs = dict(net_arch=[64, 64])
        self.multiagent_model = A2C("MlpPolicy", self.multiagent_env, gamma=0.01, verbose=0, learning_rate=0.00039,
                                    policy_kwargs=multiagent_policy_kwargs)

        # 1st increase/decrease in limit to exploit, 2nd increase/decrease in the penalty multiplier for those who exploit above the limit.
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # 1st is the resources, 2nd is short-term sustainability, 3rd is long-term sustainability
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

    def step(self, action):

        CommonsShared.limit_exploit += (action[0] * CommonsShared.max_limit_exploit_increase)

        # clipping limit_exploit between 0 and max_limit_exploit
        CommonsShared.limit_exploit = max(0, min(CommonsShared.limit_exploit,
                                                 CommonsShared.max_limit_exploit))
        #print("new limit was set to: {}".format(CommonsShared.limit_exploit))

        CommonsShared.penalty_multiplier += (
                action[1] * CommonsShared.max_penalty_multiplier_increase)

        # clipping penalty between 0 and max_penalty
        CommonsShared.penalty_multiplier = max(0, min(CommonsShared.penalty_multiplier,
                                                      CommonsShared.max_penalty_multiplier))
        #print("new penalty multiplier was set to: {}".format(CommonsShared.penalty_multiplier))

        # trains participants for n_participants_loop_steps steps
        self.multiagent_model.learn(
            total_timesteps=CommonsShared.n_agents * CommonsShared.agents_loop_steps)

        CommonsShared.period_consumption.append(sum(CommonsShared.step_consumption))
        CommonsShared.period_replenishment.append(sum(CommonsShared.step_replenishment))

        # calculate sustainability
        st_sustainability = self.calculate_sustainability(CommonsShared.n_steps_short_term)
        lt_sustainability = self.calculate_sustainability(CommonsShared.n_steps_long_term)

        state = np.array([self.normalized_resources(), st_sustainability, lt_sustainability],
                         dtype=np.float32)

        reward = self.normalized_consumption()  # reward = normalized last period consumption
        #print("regulator's reward: {}".format(reward))

        CommonsShared.periods_counter += 1
        #print(CommonsShared.periods_counter)
        #print(CommonsShared.max_episode_periods)
        done = bool(CommonsShared.periods_counter > CommonsShared.max_episode_periods
                    or CommonsShared.resources == 0)

        if done:
            Logging.log_episodes_rewards(CommonsShared.episode_number, sum(CommonsShared.period_consumption))
            CommonsShared.episode_number += 1

            # should return next_state (observation), reward, done, {}
        return state, reward, done, {}

    def reset(self):
        #print("outer env reset was called")

        CommonsShared.periods_counter = 1

        CommonsShared.resources = np.random.uniform(low=10000, high=30000)

        #print(CommonsShared.resources)

        st_sustainability = np.random.uniform(low=0.4, high=0.6)
        lt_sustainability = np.random.uniform(low=0.4, high=0.6)

        CommonsShared.period_consumption = []
        CommonsShared.period_replenishment = []

        CommonsShared.limit_exploit = np.random.normal(
            CommonsShared.max_replenishment / (CommonsShared.n_agents * 2),
            CommonsShared.max_replenishment / (8 * CommonsShared.n_agents))  # this is ugly
        CommonsShared.penalty_multiplier = np.random.normal(1, 0.2)

        state = (self.normalized_resources(), st_sustainability, lt_sustainability)

        return np.array(state, dtype=np.float32)

    def calculate_sustainability(self, n_steps_back):
        if sum(CommonsShared.period_consumption[-n_steps_back:]) != 0:
            sustainability = sum(CommonsShared.period_replenishment[-n_steps_back:]) / sum(
                CommonsShared.period_consumption[-n_steps_back:])
        else:
            sustainability = sum(
                CommonsShared.period_replenishment[-n_steps_back:]) / 0.001  # this is just so the math won't break,
                                                                    # could try to find something better later
        return sustainability / (1 + sustainability)  # returns normalized sustainability

    def normalized_resources(self):
        return CommonsShared.resources / CommonsShared.carrying_capacity

    # this works for the logistic growth function R * r * (1 - R / cc)
    def normalized_consumption(self):
        normalizing_factor = CommonsShared.max_replenishment / CommonsShared.n_agents
        return CommonsShared.period_consumption[-1] / normalizing_factor - 5  # normalizes to a -5 to 5 range
