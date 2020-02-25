from torch.multiprocessing import Pool
from contextlib import closing
import numpy as np
import torch
import cv2
from collections import deque
import  sys


class ParallelRunner(object):

    def __init__(self, n_workers, max_steps, env, agent):
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.env = env
        self.agent = agent
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.stacked_states = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)

    def run(self):
        with closing(Pool(processes=self.n_workers)) as pool:
            experiences = pool.map(self, range(self.n_workers))
            # pool.terminate()
            pool.close()
            pool.join()

        states = [experience[0] for experience in experiences]
        actions = [experience[1] for experience in experiences]
        rewards = [experience[2] for experience in experiences]
        dones = [experience[3] for experience in experiences]
        values = [experience[4] for experience in experiences]
        next_values = [experience[5] for experience in experiences]

        return states, actions, rewards, dones, values, next_values

    def __call__(self, *args, **kwargs):
        return self.run_one_episode()

    def run_one_episode(self):
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        s = self.reset_env()
        # self.env.render()
        # cv2.waitKey(0)
        state = self.stack_state(s, True)
        for _ in range(self.max_steps):
            action, v = self.agent.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action.detach())
            dones.append(done)
            values.append(v.detach())
            state = self.stack_state(next_state, False)
            self.env.render()
            # cv2.waitKey(0)
            if done:
                break

        _, next_value = self.agent.choose_action(state)
        return states, actions, rewards, dones, values, next_value.detach()

    def reset_env(self):
        seed = np.random.randint(0, sys.maxsize)
        torch.manual_seed(seed)
        s = self.env.reset()
        return s

    def stack_state(self, s, new_episode=False):
        s = self.pre_process(s)

        if new_episode:
            self.stacked_states = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)
            self.stacked_states.append(s)
            self.stacked_states.append(s)
            self.stacked_states.append(s)
            self.stacked_states.append(s)
        else:
            self.stacked_states.append(s)

        return np.stack(self.stacked_states, axis=2)

    @staticmethod
    def pre_process(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84))
        # img = np.expand_dims(img, 0)

        return img / 255.0
