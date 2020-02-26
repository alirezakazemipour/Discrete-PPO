from torch.multiprocessing import Pool
from contextlib import closing
import numpy as np
import torch
import cv2
from collections import deque
import sys
import time


class ParallelRunner(object):

    def __init__(self, n_workers, max_steps, max_episode, env, agent):
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.max_episode = max_episode
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

        states = [state for experience in experiences for state in experience[0]]
        actions = [action for experience in experiences for action in experience[1]]
        rewards = [reward for experience in experiences for reward in experience[2]]
        dones = [done for experience in experiences for done in experience[3]]
        values = [value for experience in experiences for value in experience[4]]
        next_values = [experience[5] for experience in experiences]

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), values, np.array(next_values)

    def __call__(self, *args, **kwargs):
        return self.run_one_episode()

    def run_one_episode(self):

        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        s = self.reset_env()
        state = self.stack_state(s, True)
        for _ in range(self.max_steps):
            action, v = self.agent.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action.detach().cpu().numpy())
            dones.append(done)
            values.append(v.detach().cpu().numpy())
            state = self.stack_state(next_state, False)
            # self.env.render()
            # time.sleep(0.05)
            if done:
                next_value = 0
                break
            else:
                _, next_value = self.agent.choose_action(state)
                next_value = next_value.detach().cpu().numpy()
        self.close_env()
        return states, actions, rewards, dones, np.array(values), [next_value]

    def reset_env(self):
        seed = np.random.randint(0, sys.maxsize)
        torch.manual_seed(seed)
        self.env.seed(seed)
        s = self.env.reset()
        return s

    def close_env(self):
        self.env.close()

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

        return img / 255.0
