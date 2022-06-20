import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from envs.commons.commons_shared import CommonsShared
from sd_logging.logging import Logging


def build_commons_multiagent_env():
    env = CommonsMulti()
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class CommonsMulti(AECEnv):
    metadata = {'render.modes': ['human'], "name": "commons_multiagent_v0"}

    def __init__(self):

        # participants variables
        self.possible_agents = CommonsShared.possible_agents
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # how much of the resource to consume
        self.action_spaces = {agent: Box(low=0, high=1, shape=(1,)) for agent in self.possible_agents}

        # resources
        self.observation_spaces = {agent: Box(low=0, high=1, shape=(1,)) for agent in self.possible_agents}

    def render(self, mode="human"):
        print("Current state resources: {}".format(CommonsShared.resources))
        for agent in self.agents:
            print("{} reward = {}".format(agent, self._cumulative_rewards[agent]))

    def observe(self, agent):
        return np.array(self.observations[agent])

    def close(self):
        pass

    def step(self, action):

        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection
        #print("{} is acting".format(agent))
        #print("limit_exploit is set to {}".format(CommonsShared.limit_exploit))
        #print("penalty_multiplier is set to {}".format(CommonsShared.penalty_multiplier))

        consumption = action[0] * CommonsShared.max_limit_exploit  # same as max consumption

        #print("{} resources before {} consumption".format(CommonsShared.resources, agent))
        if CommonsShared.resources - consumption < 0:
            consumption = CommonsShared.resources
            CommonsShared.resources = 0
        else:
            CommonsShared.resources -= consumption

        CommonsShared.step_consumption.append(consumption)

        #print("{} consumed {} resources".format(agent, consumption))
        #print("{} resources after {} consumption".format(CommonsShared.resources, agent))

        reward = self.normalize_consumption(consumption)
        self.rewards[agent] = reward
        #print("reward received by {} this round: {}".format(agent, reward))

        Logging.log_consumption(CommonsShared.episode_number, CommonsShared.periods_counter,
                                CommonsShared.steps_counter, agent, consumption, CommonsShared.resources,
                                CommonsShared.limit_exploit, CommonsShared.penalty_multiplier, 0, reward)

        # observe the current state
        for agent in self.agents:
            self.observations[agent] = [self.normalized_resources()]
            self.state[agent] = [self.normalized_resources()]

        #print("{} resources before replenishment".format(CommonsShared.resources))
        replenishment = 0

        # replenishes the resource after the last agent has consumed
        if self._agent_selector.is_last():
            replenishment = self.growth_function()
            CommonsShared.resources += replenishment
            CommonsShared.step_replenishment.append(replenishment)

            self._accumulate_rewards()
            self._clear_rewards()

        #print("replenishment: {}".format(replenishment))
        #print("{} resources after replenishment".format(CommonsShared.resources))
        #print("cumulative rewards: {}".format(self._cumulative_rewards))

        CommonsShared.steps_counter += 1

        # selects the next agent
        self.agent_selection = self._agent_selector.next()

    def reset(self):
        #print("inner env reset was called")

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: [self.normalized_resources()] for agent in self.agents}
        self.observations = {agent: [self.normalized_resources()] for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        CommonsShared.step_consumption = []
        CommonsShared.step_replenishment = []

    def growth_function(self):
        # returns the quantity to grow ($\delta$ R)
        growth = CommonsShared.growing_rate * CommonsShared.resources * (
                1 - CommonsShared.resources / CommonsShared.carrying_capacity)
        random_noise = np.random.normal() * growth * 0.1
        return growth + random_noise

    def normalized_resources(self):
        return CommonsShared.resources / CommonsShared.carrying_capacity

    def normalize_consumption(self, consumption):
        return consumption / (CommonsShared.max_limit_exploit / 4) - 2
