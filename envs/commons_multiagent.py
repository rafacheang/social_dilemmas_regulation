import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers, aec_to_parallel
import supersuit as ss

from envs.commons_shared import SharedEnv
from envs.utils.utils import multi_apply_penalty, multi_normalized_consumption, normalized_resources, \
    multi_growth_function, multi_normalized_limit_exploit, multi_normalized_penalty
from sd_logging.logging import Logging


def build_multiagent_env(n_agents, active_regulator=True):
    multiagent_env = CommonsMulti(n_agents, active_regulator)
    multiagent_env = wrappers.OrderEnforcingWrapper(multiagent_env)
    multiagent_env = aec_to_parallel(multiagent_env)
    multiagent_env = ss.pettingzoo_env_to_vec_env_v1(multiagent_env)
    multiagent_env = ss.concat_vec_envs_v1(multiagent_env, 1, base_class='stable_baselines3')
    return multiagent_env


class CommonsMulti(AECEnv):
    metadata = {'render.modes': ['human'],
                "name": "commons_multiagent_v0",
                "is_parallelizable": True}

    def __init__(self, n_agents, active_regulator):

        # participants variables
        self.possible_agents = ["player" + str(x) for x in range(n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # how much of the resource to consume
        self.action_spaces = {agent: Box(low=0, high=1, shape=(1,)) for agent in self.possible_agents}

        self.is_regulator_active = active_regulator
        if self.is_regulator_active:
            # 1st - max allowed to consume (limit-exploit), 2nd - penalty, 3rd - resources
            self.observation_spaces = {agent: Box(low=0, high=1, shape=(3,)) for agent in self.possible_agents}
        else:
            # resources (no need to observe allowed consumption or penalty if there is no regulator)
            self.observation_spaces = {agent: Box(low=0, high=1, shape=(1,)) for agent in self.possible_agents}

        # --- # --- # --- # --- # --- # --- # SIMULATION CONSTANTS # --- # --- # --- # --- # --- # --- #

        self.n_agents = n_agents

        # episode counter is initialized in this env in case regulator is not active
        # we also define an upper bound to number of steps per episode in case regulator is inactive
        if not self.is_regulator_active:
            self.max_steps_per_episode = 200

        # --- # --- # --- # --- # --- # --- # END OF SIMULATION CONSTANTS # --- # --- # --- # --- # --- # --- #

    def render(self, mode="human"):
        print("Current state resources: {}".format(SharedEnv.resources))
        for agent in self.agents:
            print("{} reward = {}".format(agent, self._cumulative_rewards[agent]))

    def observe(self, agent):
        return np.array(self.observations[agent])

    def close(self):
        pass

    def step(self, action):

        if self.dones[self.agent_selection]:
            return self._was_done_step(None)

        agent = self.agent_selection
        # print("{} is acting".format(agent))
        # print("limit_exploit is set to {}".format(SharedEnv.limit_exploit))
        # print("penalty_multiplier is set to {}".format(SharedEnv.penalty_multiplier))

        consumption = action[0] * SharedEnv.max_consumption

        # print("{} resources before {} consumption".format(SharedEnv.resources, agent))
        if SharedEnv.resources - consumption < 0:
            consumption = SharedEnv.resources
            SharedEnv.resources = 0
        else:
            SharedEnv.resources -= consumption

        SharedEnv.period_consumption.append(consumption)

        # print("{} consumed {} resources".format(agent, consumption))
        # print("{} resources after {} consumption".format(SharedEnv.resources, agent))

        penalty = 0
        if self.is_regulator_active:
            penalty = multi_apply_penalty(consumption, SharedEnv.limit_exploit,
                                          SharedEnv.penalty_multiplier,
                                          SharedEnv.punishment_probability)
            # print("applied penalty was {}".format(penalty))

        reward = multi_normalized_consumption(consumption - penalty, SharedEnv.max_consumption)
        self.rewards[agent] = reward
        # print("reward received by {} this round: {}".format(agent, reward))

        if self.is_regulator_active:
            Logging.log_consumption(SharedEnv.episode_number, SharedEnv.periods_counter,
                                    self.step_counter, agent, consumption, SharedEnv.resources,
                                    SharedEnv.limit_exploit, SharedEnv.penalty_multiplier, penalty, reward)
        else:
            Logging.log_consumption(SharedEnv.episode_number, self.step_counter, agent,
                                    consumption, SharedEnv.resources, reward)

        # observe the current state
        for agent in self.agents:

            if self.is_regulator_active:
                self.observations[agent] = [multi_normalized_limit_exploit(SharedEnv.limit_exploit, SharedEnv.max_consumption),
                                            multi_normalized_penalty(SharedEnv.penalty_multiplier, SharedEnv.max_penalty_multiplier),
                                            normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)]
                self.state[agent] = [multi_normalized_limit_exploit(SharedEnv.limit_exploit, SharedEnv.max_consumption),
                                     multi_normalized_penalty(SharedEnv.penalty_multiplier, SharedEnv.max_penalty_multiplier),
                                     normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)]

            else:
                self.observations[agent] = [normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)]
                self.state[agent] = [normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)]

        # print("{} resources before replenishment".format(SharedEnv.resources))
        replenishment = 0

        # replenishes the resource after the last agent has consumed
        if self._agent_selector.is_last():
            replenishment = multi_growth_function(SharedEnv.growth_rate, SharedEnv.resources, SharedEnv.carrying_capacity)
            SharedEnv.resources += replenishment
            SharedEnv.period_replenishment.append(replenishment)

            self._accumulate_rewards()
            self._clear_rewards()

            if not self.is_regulator_active:

                if SharedEnv.resources == 0 or self.step_counter >= self.max_steps_per_episode:
                    Logging.log_episodes_rewards(SharedEnv.episode_number, sum(SharedEnv.period_consumption))
                    self.dones = {agent: True for agent in self.agents}
                    # print('dones were set to True')

        # print("replenishment: {}".format(replenishment))
        # print("{} resources after replenishment".format(SharedEnv.resources))
        # print("cumulative rewards: {}".format(self._cumulative_rewards))

        self.step_counter += 1

        # selects the next agent
        self.agent_selection = self._agent_selector.next()

    def reset(self, seed=None, options=None):
        # print("inner env reset was called")

        # step counter
        self.step_counter = 1

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        if self.is_regulator_active:
            self.state = {
                agent: [multi_normalized_limit_exploit(SharedEnv.limit_exploit, SharedEnv.max_consumption),
                        multi_normalized_penalty(SharedEnv.penalty_multiplier, SharedEnv.max_penalty_multiplier),
                        normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)]
                for agent in self.agents}
            self.observations = {
                agent: [multi_normalized_limit_exploit(SharedEnv.limit_exploit, SharedEnv.max_consumption),
                        multi_normalized_penalty(SharedEnv.penalty_multiplier, SharedEnv.max_penalty_multiplier),
                        normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)]
                for agent in self.agents}

        else:
            # we count episodes in this env in case regulator is inactive
            SharedEnv.episode_number += 1

            # max consumption per agent
            SharedEnv.max_consumption = SharedEnv.max_replenishment * 2 / self.n_agents  # two times max replenishment per agent

            SharedEnv.resources = np.random.uniform(low=10000, high=30000)
            self.state = {agent: [normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)] for agent in self.agents}
            self.observations = {agent: [normalized_resources(SharedEnv.resources, SharedEnv.carrying_capacity)] for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # consumption and replenishment buffers
        SharedEnv.period_consumption = []
        SharedEnv.period_replenishment = []