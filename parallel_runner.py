from torch.multiprocessing import Pool
from contextlib import closing
import numpy as np
import torch


class ParallelRunner(object):

    def __init__(self, n_workers, max_steps, env, agent):
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.env = env
        self.agent = agent

    def run(self):
        with closing(Pool(processes=self.n_workers)) as pool:
            experiences = pool.map(self, range(self.n_workers))
            pool.terminate()

        states = [experience[0] for experience in experiences]
        actions = [experience[1] for experience in experiences]
        rewards = [experience[2] for experience in experiences]
        dones = [experience[3] for experience in experiences]
        values = [experience[4] for experience in experiences]
        next_values = [experience[5] for experience in experiences]

        return states, actions, rewards, dones, values, next_values

    def __call__(self, *args, **kwargs):
        self.run_one_episode()

    def run_one_episode(self):
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        s = self.reset_env()
        for _ in self.max_steps:
            action, v = self.agent.choose_action(s)
            next_state, reward, done, _ = self.env.step(action)
            states.append(s)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            values.append(v)
            s = next_state
        _, next_value = self.agent.choose_action(next_state)

        return states, actions, rewards, dones, values, next_value

    def reset_env(self):
        seed = np.random.randint(0, np.inf)
        torch.manual_seed(seed)
        s = self.env.reset()
        return s
