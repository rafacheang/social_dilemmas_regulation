class Logging:
    episodes_rewards = []
    consumption = []

    @classmethod
    def log_episodes_rewards(cls, episode_number, episode_reward):
        cls.episodes_rewards.append((episode_number, episode_reward))

    @classmethod
    def log_consumption(cls, episode_number, period_number, step_number, agent_name,
                        consumption, resources, limit_exploit, penalty_multiplier, penalty, reward):
        cls.consumption.append((episode_number, period_number, step_number, agent_name, consumption,
                                resources, limit_exploit, penalty_multiplier, penalty, reward))
