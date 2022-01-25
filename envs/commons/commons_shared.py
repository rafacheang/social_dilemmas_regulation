class CommonsShared:
    # period, step, and episode counter
    periods_counter = 1
    steps_counter = 1
    episode_number = 0

    # number of periods considered for calculating short and long term sustainabilities
    n_steps_short_term = 1
    n_steps_long_term = 4

    # number of agents
    n_agents = 5

    # max periods per episode
    max_episode_periods = 10

    # rounds played by each agent in a period
    agents_loop_steps = 20

    # resources growing rate and environment carrying capacity
    growing_rate = 0.3  # Ghorbani uses 0.25 - 0.35
    carrying_capacity = 50000  # Ghorbani uses 10000 - 20000

    # probability of being caught wrong-doing
    punishment_probability = 1

    # max replenishment (dependent on growth function)
    max_replenishment = carrying_capacity * growing_rate * 0.5 ** 2

    # max consumption, penalty, and increases in consumption and penalty
    max_penalty_multiplier = 3
    max_penalty_multiplier_increase = 0.5
    max_limit_exploit = max_replenishment * 2 / n_agents  # two times max replenishment
    max_limit_exploit_increase = 1000

    #consumption and replenishment buffers
    consumed_buffer = []
    consumed = []
    replenished_buffer = []
    replenished = []
