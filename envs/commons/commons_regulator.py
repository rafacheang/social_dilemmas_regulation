import gym
import numpy as np
import supersuit as ss
from gym.spaces import Box
from pettingzoo.utils import to_parallel
from stable_baselines3 import A2C

from envs.commons.commons_multiagent import build_commons_multiagent_env


class CommonsRegulator(gym.Env):

    def __init__(self, shared_env):

        # instance of shared environment
        self.commons_shared_env = shared_env

        self.periods_counter = 1

        # number of periods considered for calculating short and long term sustainabilities
        self.n_steps_short_term = 1
        self.n_steps_long_term = 5

        # multiagent environment
        self.multiagent_env = build_commons_multiagent_env(shared_env)
        self.multiagent_env = to_parallel(self.multiagent_env)
        self.multiagent_env = ss.pettingzoo_env_to_vec_env_v1(self.multiagent_env)
        self.multiagent_env = ss.concat_vec_envs_v1(self.multiagent_env, 1, base_class='stable_baselines3')

        # multiagent model
        multiagent_policy_kwargs = dict(net_arch=[64, 64])
        self.multiagent_model = A2C("MlpPolicy", self.multiagent_env, gamma=0, verbose=0, learning_rate=0.0004,
                                    policy_kwargs=multiagent_policy_kwargs)

        # 1st increase/decrease in limit to exploit, 2nd increase/decrease in the penalty multiplier for those who exploit above the limit.
        # Both in percentage
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # 1st is the resources, 2nd is short-term sustainability, 3rd is long-term sustainability
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

    def step(self, action):

        self.commons_shared_env.limit_exploit += (action[0] * self.commons_shared_env.max_limit_exploit_increase)

        # clipping limit_exploit between 0 and max_limit_exploit
        self.commons_shared_env.limit_exploit = max(0, min(self.commons_shared_env.limit_exploit,
                                                           self.commons_shared_env.max_limit_exploit))

        self.commons_shared_env.penalty_multiplier += (
                    action[1] * self.commons_shared_env.max_penalty_multiplier_increase)

        # clipping penalty between 0 and max_penalty
        self.commons_shared_env.penalty_multiplier = max(0, min(self.commons_shared_env.penalty_multiplier,
                                                                self.commons_shared_env.max_penalty_multiplier))

        # Logging.log_actions(self.commons_shared_env.episode_number, action[0] * self.commons_shared_env.max_limit_exploit_increase,
        #                    action[1] * self.commons_shared_env.max_penalty_multiplier_increase)
        # Logging.log_limit_exploit(self.commons_shared_env.episode_number, self.commons_shared_env.limit_exploit)
        # Logging.log_penalty_multiplier(self.commons_shared_env.episode_number, self.commons_shared_env.penalty_multiplier)

        # trains participants for n_participants_loop_steps steps
        self.multiagent_model.learn(total_timesteps=self.commons_shared_env.n_agents * self.commons_shared_env.agents_loop_steps)

        self.commons_shared_env.consumed.append(sum(self.commons_shared_env.consumed_buffer))
        self.commons_shared_env.replenished.append(sum(self.commons_shared_env.replenished_buffer))

        # calculate sustainability
        st_sustainability = self.calculate_sustainability(self.n_steps_short_term)
        lt_sustainability = self.calculate_sustainability(self.n_steps_long_term)

        self.periods_counter += 1

        state = np.array([self.normalized_resources(), st_sustainability, lt_sustainability],
                              dtype=np.float32)

        # Logging.log_states(self.commons_shared_env.episode_number, self.state)
        # Logging.log_steps_consumption(self.commons_shared_env.episode_number, self.commons_shared_env.consumed[-1])

        reward = self.normalized_consumption()  # reward = last period consumption normalized

        done = bool(self.periods_counter > self.commons_shared_env.max_episode_timestep - 1
                    or self.commons_shared_env.resources == 0)

        self.periods_counter += 1

        if done:
            # Logging.log_episodes_rewards(self.commons_shared_env.episode_number, sum(self.commons_shared_env.consumed))
            self.commons_shared_env.episode_number += 1

            # should return next_state (observation), reward, done, {}
        return state, reward, done, {}

    def reset(self):
        # print("outer env reset was called")

        self.commons_shared_env.resources = np.random.uniform(low=10000, high=30000)
        st_sustainability = np.random.uniform(low=0.4, high=0.6)
        lt_sustainability = np.random.uniform(low=0.4, high=0.6)

        self.commons_shared_env.consumed = []
        self.commons_shared_env.replenished = []

        self.commons_shared_env.limit_exploit = np.random.normal(
            self.commons_shared_env.max_replenishment / self.commons_shared_env.n_agents,
            self.commons_shared_env.max_replenishment / (8 * self.commons_shared_env.n_agents)) #this is ugly
        self.commons_shared_env.penalty_multiplier = np.random.normal(1, 0.2)

        state = (self.normalized_resources(), st_sustainability, lt_sustainability)
        # Logging.log_states(self.commons_shared_env.episode_number, self.state)
        return np.array(state, dtype=np.float32)

    def calculate_sustainability(self, n_steps_back):
        if sum(self.commons_shared_env.consumed[-n_steps_back:]) != 0:
            sustainability = sum(self.commons_shared_env.replenished[-n_steps_back:]) / sum(
                self.commons_shared_env.consumed[-n_steps_back:])
        else:
            sustainability = sum(
                self.commons_shared_env.replenished[-n_steps_back:]) / 0.001  # this is just so the math won't break,
            # should find something better later
        return sustainability / (1 + sustainability)  # returns normalized sustainability

    def normalized_resources(self):
        return self.commons_shared_env.resources / self.commons_shared_env.carrying_capacity

    # this works for the logistic growth function R * r * (1 - R / cc)
    def normalized_consumption(self):
        normalizing_factor = self.commons_shared_env.max_replenishment / self.commons_shared_env.n_agents
        return self.commons_shared_env.consumed[-1] / normalizing_factor - 5  # normalizes to a -5 to 5 range
