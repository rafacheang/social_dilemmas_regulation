class Logging:
    episodes_rewards = []
    consumption = []

    @classmethod
    def log_episodes_rewards(cls, episode_number, episode_reward):
        cls.episodes_rewards.append((episode_number, episode_reward))

    @classmethod
    def log_consumption(cls, *args):
        cls.consumption.append(args)
