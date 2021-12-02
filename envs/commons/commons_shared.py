from envs.shared_env_base import SharedEnvBase


class CommonsShared(SharedEnvBase):
    def __init__(self):
        SharedEnvBase.__init__(self)

        # ---#---# ENVIRONMENT CONSTANTS #---#---#

        # resources growing rate and environment carrying capacity
        self.growing_rate = 0.3  # Ghorbani uses 0.25 - 0.35
        self.carrying_capacity = 50000  # Ghorbani uses 10000 - 20000

        # probability of being caught wrong-doing
        self.punishment_probability = 1

        # max replenishment (dependent on growth function)
        self.max_replenishment = self.carrying_capacity * self.growing_rate * 0.5 ** 2

        # max consumption, penalty and increseases in consumption and penalty
        self.max_penalty_multiplier = 2
        self.max_penalty_multiplier_increase = 0.2
        self.penalty_multiplier = 0

        self.max_limit_exploit = self.max_replenishment * 2 / self.n_agents  # two times max replenishment
        self.max_limit_exploit_increase = 1000
        self.limit_exploit = 0

        self.resources = 10000

        # ---#---# ENVIRONMENT VARIABLES #---#---#

        self.consumed_buffer = []
        self.consumed = []
        self.replenished_buffer = []
        self.replenished = []