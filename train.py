from parallel_runner import ParallelRunner
from torch.distributions import Categorical
import torch
import numpy as np
from torch import multiprocessing as mp


class Train:
    def __init__(self, env, agent, n_workers, max_steps, epochs, mini_batch_size, epsilon):
        self.env = env
        self.agent = agent
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.parallel_runner = ParallelRunner(self.n_workers, self.max_steps, self.env, self.agent)
        self.epsilon = epsilon
        self.episode_counter = 0
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        mp.set_start_method('spawn')


    @staticmethod
    def choose_mini_batch(self, mini_batch_size, states, actions, returns, advs):
        full_batch_size = states.shape[0]
        for _ in range(full_batch_size // mini_batch_size):
            idxes = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[idxes:], actions[idxes:], returns[:], advs[:]

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_gae(rewards, values, next_values, dones)
        advs = returns - values

        for epoch in range(self.epochs):
            for state, action, q_value, adv in self.choose_mini_batch(self.mini_batch_size,
                                                                      states, actions, returns, advs):
                dist, value = self.agent.new_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.agent.new_policy, state, action)
                old_log_prob = self.calculate_log_probs(self.agent.old_policy, state, action)
                ratio = torch.exp(new_log_prob) / (torch.exp(old_log_prob) + 1e-8)

                actor_loss = self.compute_ac_loss(ratio, adv)
                crtitic_loss = self.agent.critic_loss(q_value - value)

                total_loss = 0. * crtitic_loss + actor_loss - 0.001 * entropy
                self.agent.optimize(total_loss)

    def equalize_policies(self):
        self.agent.set_weights()

    def step(self):
        states, actions, rewards, dones, values, next_values = self.parallel_runner.run()
        self.episode_counter += 1

        self.train(states, actions, rewards, dones, values, next_values)
        self.equalize_policies()

    def get_gae(self, rewards, values, next_values, dones, gamma=0.99, lamda=0.95):
        values = values + [next_values]
        gae = 0
        returns = []
        for step in (reversed(rewards)):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lamda * (1 - dones[step]) * gae
            returns.insert(0, gae + [values])

        return returns

    def calculate_ratio(self, states, actions):
        new_policy_log = self.calculate_log_probs(self.agent.new_policy, states, actions)
        old_policy_log = self.calculate_log_probs(self.agent.old_policy, states, actions)
        ratio = torch.exp(new_policy_log) / (torch.exp(old_policy_log) + 1e-8)
        return ratio

    @staticmethod
    def calculate_log_probs(self, model, states, actions):
        policy_output, _ = model(states)
        policy_distribution = Categorical(policy_output)
        return policy_distribution.log_prob(actions)

    def compute_ac_loss(self, ratio, adv):
        r_new = ratio * adv
        clamped_r = torch.clamp(r_new, 1 - self.epsilon, 1 + self.epsilon)
        loss = min(r_new, clamped_r)
        loss = - loss.mean()
        return loss
