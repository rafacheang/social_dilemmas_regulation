class SharedEnv:

    episode_number = 0

    # periods_counter is initialized at regulator's env reset
    # periods_counter

    # one of the other two envs will initialize resources in their reset() method
    # resources

    carrying_capacity = 50000
    growth_rate = 0.3

    # probability of being caught wrong-doing
    punishment_probability = 1

    # max replenishment (dependent on growth function)
    max_replenishment = carrying_capacity * growth_rate * 0.5 ** 2

    # the regulator's env will initialize limit_exploit and penalty multiplier in reset()
    # limit_exploit
    # penalty_multiplier
    max_penalty_multiplier = 3

    # max_consumption is initialized within the multiagent_env's reset method
    # max_consumption