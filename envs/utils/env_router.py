from envs.commons_multiagent import build_multiagent_env
from envs.commons_regulator import build_regulator_env


def build_env(n_agents=5, active_regulator=True, penalty_multiplier=None):

    multiagent_env = build_multiagent_env(n_agents, active_regulator)
    if active_regulator:
        return build_regulator_env(multiagent_env, n_agents, penalty_multiplier)

    else:
        return multiagent_env
