from parallel_runner import ParallelRunner


class Train:
    def __init__(self, env, agent, n_workers, max_steps):
        self.env = env
        self.agent = agent
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.parallel_runner = ParallelRunner(self.n_workers, self.max_steps, self.env, self.agent)

        self.episode_counter = 0

    def rgb_to_gray(self, img):
        pass

    def train(self, states, actions, rewards, dones, values):

        returns = self.get_gae(rewards, values)

    def equalize_policies(self):
        pass

    def step(self):
        states, actions, rewards, dones = self.parallel_runner.run()
        self.episode_counter += 1

        self.train(states, actions, rewards, dones)
        self.equalize_policies()

    def get_gae(self, r, v):
        return 0
