class SharedEnvBase:
    def __init__(self):
        # number of agents
        self.n_agents = 2

        # max timesteps per episode
        self.max_episode_timestep = 10

        # rounds played by each agent in a period
        self.agents_loop_steps = 10
        #self.n_participants_loop_steps = self.agents_loop_steps * self.n_agents

        self.episode_number = 0