import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob
from collections import deque


class Logger:
    def __init__(self, brain, **config):
        self.config = config
        self.brain = brain
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_reward = 0
        self.running_reward = 0
        self.running_act_prob = 0
        self.position = 0
        self.running_position = 0
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.running_training_logs = 0
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.running_last_10_r = 0  # It is not correct but does not matter.

        if self.config["do_train"] and self.config["train_from_scratch"]:
            self.create_wights_folder()
            self.log_params()

        self.exp_avg = lambda x, y: 0.9 * x + 0.1 * y if (y != 0).all() else y

    def create_wights_folder(self):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + self.log_dir)

    def log_params(self):
        with SummaryWriter("Logs/" + self.log_dir) as writer:
            for k, v in self.config.items():
                writer.add_text(k, str(v))

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, *args):
        iteration, training_logs, action_prob = args

        self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        self.running_training_logs = self.exp_avg(self.running_training_logs, np.array(training_logs))

        if iteration % (self.config["interval"] // 3) == 0:
            self.save_params(self.episode, iteration)

        with SummaryWriter("Logs/" + self.log_dir) as writer:
            writer.add_scalar("Episode Reward", self.episode_reward, self.episode)
            writer.add_scalar("Running Episode Reward", self.running_reward, self.episode)
            writer.add_scalar("Position", self.position, self.episode)
            writer.add_scalar("Running last 10 Reward", self.running_last_10_r, self.episode)
            writer.add_scalar("Max Episode Reward", self.max_episode_reward, self.episode)
            writer.add_scalar("Running Action Probability", self.running_act_prob, iteration)
            writer.add_scalar("Running Position", self.running_position, iteration)
            writer.add_scalar("Running PG Loss", self.running_training_logs[0], iteration)
            writer.add_scalar("Running Value Loss", self.running_training_logs[1], iteration)
            writer.add_scalar("Running Entropy", self.running_training_logs[2], iteration)
            writer.add_scalar("Running Explained variance", self.running_training_logs[3], iteration)

        self.off()
        if iteration % self.config["interval"] == 0:
            print("Iter: {}| "
                  "E: {}| "
                  "E_Reward: {:.1f}| "
                  "E_Running_Reward: {:.1f}| "
                  "Position: {}| "
                  "Running Position: {:.1f}| "
                  "LR: {}| "
                  "Clip Range: {:.3f}| "
                  "Iter_Duration: {:.3f}| "
                  "Time: {} "
                  .format(iteration,
                          self.episode,
                          self.episode_reward,
                          self.running_reward,
                          self.position,
                          self.running_position,
                          self.brain.scheduler.get_last_lr(),
                          self.brain.epsilon,
                          self.duration,
                          datetime.datetime.now().strftime("%H:%M:%S"),
                          )
                  )
        self.on()

    def log_episode(self, *args):
        self.episode, self.episode_reward, self.position = args

        self.max_episode_reward = max(self.max_episode_reward, self.episode_reward)

        self.running_reward = self.exp_avg(self.running_reward, self.episode_reward)
        self.running_position = self.exp_avg(self.running_position, self.position)

        self.last_10_ep_rewards.append(self.episode_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    def save_params(self, episode, iteration):
        torch.save({"current_policy_state_dict": self.brain.current_policy.state_dict(),
                    "optimizer_state_dict": self.brain.optimizer.state_dict(),
                    "scheduler_state_dict": self.brain.scheduler.state_dict(),
                    "iteration": iteration,
                    "episode": episode,
                    "running_reward": self.running_reward,
                    "position": self.position
                    },
                   "Models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        return checkpoint
