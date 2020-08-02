import torch
from torch import device
import time
import os
from utils import *


class Play:
    def __init__(self, env, agent, max_episode=10):
        self.env = make_atari(env)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + "ppo_pong" + ".avi", self.fourcc, 50.0,
                                           self.env.observation_space.shape[1::-1])

    def evaluate(self):
        stacked_states = np.zeros((84, 84, 4), dtype=np.uint8)
        for ep in range(self.max_episode):
            self.env.seed(ep)
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            # x = input("Push any button to proceed...")
            for _ in range(self.env._max_episode_steps):
                action, _ = self.agent.get_actions_and_values(stacked_states)
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                stacked_states = stack_states(stacked_states, s_, False)
                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward:3.3f}")
            self.env.close()
            self.VideoWriter.release()
            cv2.destroyAllWindows()
