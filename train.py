from parallel_runner import ParallelRunner
from copy import deepcopy
import torch
import numpy as np
from torch import multiprocessing as mp
from test import evaluate_model
from copy import deepcopy


class Train:
    def __init__(self, env, agent, n_workers, max_steps, max_episode, epochs, mini_batch_size, epsilon):
        self.env = env
        self.agent = agent
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.max_episode = max_episode
        self.parallel_runner = ParallelRunner(self.n_workers, self.max_steps, self.max_episode, self.env, self.agent)
        self.epsilon = epsilon
        self.episode_counter = 0
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        torch.cuda.empty_cache()

        mp.set_start_method('spawn')

        self.global_running_r = []

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            idxes = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[idxes], actions[idxes], returns[idxes], advs[idxes]

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_gae(rewards, deepcopy(values), next_values, dones)
        advs = returns - np.vstack(values).reshape((sum([len(values[i]) for i in range(self.n_workers)]), 1))
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for epoch in range(self.epochs):
            for state, action, q_value, adv in self.choose_mini_batch(self.mini_batch_size,
                                                                      states, actions, returns, advs):
                state = torch.Tensor(state).permute([0, 3, 1, 2]).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                q_value = torch.Tensor(q_value).to(self.agent.device).view((self.mini_batch_size, 1))

                dist, value = self.agent.new_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.agent.new_policy, state, action)
                old_log_prob = self.calculate_log_probs(self.agent.old_policy, state, action)
                # ratio = torch.exp(new_log_prob) / (torch.exp(old_log_prob) + 1e-8)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_ac_loss(ratio, adv)
                # crtitic_loss = self.agent.critic_loss(q_value, value)
                critic_loss = (q_value - value).pow(2).mean()

                total_loss = 1.0 * critic_loss + actor_loss - 0.01 * entropy
                self.agent.optimize(total_loss)

                return total_loss, entropy, rewards

    def equalize_policies(self):
        self.agent.set_weights()

    def step(self):

        while self.episode_counter < self.max_episode:
            states, actions, rewards, dones, values, next_values = self.parallel_runner.run()
            self.episode_counter += 1

            total_loss, entropy, rewards = self.train(states, actions, rewards, dones, values, next_values)
            self.equalize_policies()
            if self.episode_counter % 10 == 0:
                evaluation_rewards = evaluate_model(self.agent, deepcopy(self.env))
                self.print_logs(total_loss, entropy, evaluation_rewards)
        self.agent.save_weights()

    def get_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):

        returns = [[] for _ in range(self.n_workers)]

        for worker in range(self.n_workers):
            values[worker] = values[worker] + next_values[worker]
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + gamma * (values[worker][step + 1]) * (1 - dones[worker][step]) - values[worker][step]
                gae = delta + gamma * lam * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + values[worker][step])

        return np.vstack(returns).reshape((sum([len(returns[i]) for i in range(self.n_workers)]), 1))

    def calculate_ratio(self, states, actions):
        new_policy_log = self.calculate_log_probs(self.agent.new_policy, states, actions)
        old_policy_log = self.calculate_log_probs(self.agent.old_policy, states, actions)
        ratio = torch.exp(new_policy_log) / (torch.exp(old_policy_log) + 1e-8)
        return ratio

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution, _ = model(states)
        return policy_distribution.log_prob(actions)

    def compute_ac_loss(self, ratio, adv):
        r_new = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(r_new, clamped_r)
        loss = - loss.mean()
        return loss

    def print_logs(self, total_loss, entropy, rewards):

        if self.episode_counter == 10:
            self.global_running_r.append(rewards)
        else:
            self.global_running_r.append(self.global_running_r[-1] * 0.99 + rewards * 0.01)

        print(f"Ep:{self.episode_counter}| "
              f"Ep_Reward:{rewards}| "
              f"Running_reward:{self.global_running_r[-1]:3.3f}| "
              f"Total_loss:{total_loss:3.3f}| "
              f"Entropy:{entropy:3.3f}| ")
