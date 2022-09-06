import numpy as np


# sustainability calculation for the regulator's environment
def regulator_calculate_sustainability(episode_consumption_buffer, episode_replenishment_buffer, n_steps_back):
    if sum(episode_consumption_buffer[-n_steps_back:]) != 0:
        sustainability = sum(episode_replenishment_buffer[-n_steps_back:]) / sum(
            episode_consumption_buffer[-n_steps_back:])
    else:
        sustainability = sum(
            episode_replenishment_buffer[-n_steps_back:]) / 0.001  # this is just so the math won't break,
                                                                  # could try to find something better later
    return sustainability / (1 + sustainability)  # returns normalized sustainability


# normalizing functions for the regulator's environment
# this works for the logistic growth function R * r * (1 - R / cc)
def regulator_normalized_consumption(max_replenishment, n_agents, period_consumption_buffer):
    normalizing_factor = max_replenishment / n_agents
    return period_consumption_buffer[-1] / normalizing_factor - 5  # normalizes to a -5 to 5 range


# normalizing functions for the multiagent environment
def multi_normalized_limit_exploit(limit_exploit, max_consumption):
    return limit_exploit / max_consumption


def multi_normalized_penalty(penalty_multiplier, max_penalty_multiplier):
    return penalty_multiplier / max_penalty_multiplier


def multi_normalized_consumption(consumption, max_consumption):
    return consumption / (max_consumption / 4) - 2


# normalizing functions used in both environments
def normalized_resources(resources, carrying_capacity):
    return resources / carrying_capacity


# growth function for the multiagent environment
def multi_growth_function(growth_rate, resources, carrying_capacity):
    # returns the quantity to grow ($\delta$ R)
    growth = growth_rate * resources * (1 - resources / carrying_capacity)
    random_noise = np.random.normal() * growth * 0.1
    return growth + random_noise


# penalty calculation for the multiagent environment
def multi_apply_penalty(consumption, limit_exploit, penalty_multiplier, punishment_prob):
    if consumption > limit_exploit:
        if np.random.uniform() <= punishment_prob:
            return multi_calculate_penalty(consumption, limit_exploit, penalty_multiplier)
    return 0


def multi_calculate_penalty(consumption, limit_exploit, penalty_multiplier):
    return (consumption - limit_exploit) * (penalty_multiplier + 1)
