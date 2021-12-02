import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def build_commons_multiagent_env(shared_env):
    env = CommonsMulti(shared_env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class CommonsMulti(AECEnv):
    metadata = {'render.modes': ['human'], "name": "commons_multiagent_v0"}

    def __init__(self, shared_env):

        # instance of shared environment
        self.commons_shared_env = shared_env

        # participants variables
        self.possible_agents = ["player" + str(x) for x in range(shared_env.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # how much of the resource to consume
        self.action_spaces = {agent: Box(low=0, high=1, shape=(1,)) for agent in self.possible_agents}

        # 1st - max allowed to consume (limit-exploit), 2nd - penalty, 3rd - resources
        self.observation_spaces = {agent: Box(low=0, high=1, shape=(3,)) for agent in self.possible_agents}

    def render(self, mode="human"):
        print("Current state resources: {}".format(self.resources))
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
        # print("{} is acting".format(agent))
        # print("limit_exploit is set to {}".format(Environment.limit_exploit))
        # print("penalty_multiplier is set to {}".format(Environment.penalty_multiplier))

        consumption = action[0] * self.commons_shared_env.max_limit_exploit

        # print("{} resources before {} consumption".format(Environment.resources, agent))
        if self.commons_shared_env.resources - consumption < 0:
            consumption = self.commons_shared_env.resources
            self.commons_shared_env.resources = 0
        else:
            self.commons_shared_env.resources -= consumption

        self.commons_shared_env.consumed_buffer.append(consumption)
        # print("{} consumed {} resources".format(agent, consumption))
        # print("{} resources after {} consumption".format(Environment.resources, agent))

        penalty = self.apply_penalty(consumption)
        # print("applied penalty was {}".format(penalty))

        reward = self.normalize_consumption(consumption - penalty)
        self.rewards[agent] = reward
        # print("reward received by {} this round: {}".format(agent, reward))

        # observe the current state
        for agent in self.agents:
            self.observations[agent] = [self.normalized_limit_exploit(), self.normalized_penalty(),
                                        self.normalized_resources()]
            self.state[agent] = [self.normalized_limit_exploit(), self.normalized_penalty(),
                                 self.normalized_resources()]

        # print("{} resources before replenishment".format(self.commons_shared_env.resources))
        replenishment = 0
        # replenishes the resource after the last agent has consumed
        if self._agent_selector.is_last():
            replenishment = self.growth_function()
            self.commons_shared_env.resources += replenishment
            self.commons_shared_env.replenished_buffer.append(replenishment)

            self._accumulate_rewards()
            self._clear_rewards()

        # print("replenishment: {}".format(replenishment))
        # print("{} resources after replenishment".format(Environment.resources))
        # print("cumulative rewards: {}".format(self._cumulative_rewards))

        # Logging.log_participants_loop(Environment.episode_number, consumption, replenishment)
        # Logging.log_participants_rewards(Environment.episode_number, reward)
        # Logging.log_applied_penalty(Environment.episode_number, penalty)

        # selects the next agent
        self.agent_selection = self._agent_selector.next()

    def reset(self):
        # print("inner env reset was called")

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: [self.normalized_limit_exploit(), self.normalized_penalty(), self.normalized_resources()]
                      for agent in self.agents}
        self.observations = {
            agent: [self.normalized_limit_exploit(), self.normalized_penalty(), self.normalized_resources()] for agent
            in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.commons_shared_env.consumed_buffer = []
        self.commons_shared_env.replenished_buffer = []

    def growth_function(self):
        # returns the quantity to grow ($\delta$ R)
        growth = self.commons_shared_env.growing_rate * self.commons_shared_env.resources * (
                1 - self.commons_shared_env.resources / self.commons_shared_env.carrying_capacity)
        random_noise = np.random.normal() * growth * 0.1
        return growth + random_noise

    def normalized_limit_exploit(self):
        return self.commons_shared_env.limit_exploit / self.commons_shared_env.max_limit_exploit

    def normalized_penalty(self):
        return self.commons_shared_env.penalty_multiplier / self.commons_shared_env.max_penalty_multiplier

    def normalized_resources(self):
        return self.commons_shared_env.resources / self.commons_shared_env.carrying_capacity

    def normalize_consumption(self, consumption):
        return consumption / (self.commons_shared_env.max_limit_exploit / 2) - 1

    def apply_penalty(self, consumption):
        if consumption > self.commons_shared_env.limit_exploit:
            if np.random.uniform() < self.commons_shared_env.punishment_probability:
                return self.calculate_penalty(consumption)
        return 0

    def calculate_penalty(self, consumption):
        return consumption + (
                    consumption - self.commons_shared_env.limit_exploit) * self.commons_shared_env.penalty_multiplier