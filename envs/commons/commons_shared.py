class CommonsShared:

    periods_counter = 1

    # number of periods considered for calculating short and long term sustainabilities
    n_steps_short_term = 1
    n_steps_long_term = 5

    # number of agents
    n_agents = 2

    # max periods per episode
    max_episode_periods = 3

    # rounds played by each agent in a period
    agents_loop_steps = 5
    # self.n_participants_loop_steps = self.agents_loop_steps * self.n_agents

    # episode count
    episode_number = 0

    # resources growing rate and environment carrying capacity
    growing_rate = 0.3  # Ghorbani uses 0.25 - 0.35
    carrying_capacity = 50000  # Ghorbani uses 10000 - 20000

    # probability of being caught wrong-doing
    punishment_probability = 1

    # max replenishment (dependent on growth function)
    max_replenishment = carrying_capacity * growing_rate * 0.5 ** 2

    # max consumption, penalty and increseases in consumption and penalty
    max_penalty_multiplier = 2
    max_penalty_multiplier_increase = 0.2
    penalty_multiplier = 0

    max_limit_exploit = max_replenishment * 2 / n_agents  # two times max replenishment
    max_limit_exploit_increase = 1000
    limit_exploit = 0

    resources = 10000

    # ---#---# ENVIRONMENT VARIABLES #---#---#

    consumed_buffer = []
    consumed = []
    replenished_buffer = []
    replenished = []