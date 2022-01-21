class Logging:
    episodes_rewards = []

    @classmethod
    def log_episodes_rewards(cls, episode_number, episode_reward):
        cls.episodes_rewards.append((episode_number, episode_reward))