from model import Model
import torch
from torch import from_numpy
import numpy as np
from torch.optim.adam import Adam


class Brain:
    def __init__(self, state_shape, n_actions, device, n_workers, epochs, epsilon, lr):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.n_workers = n_workers
        self.mini_batch_size = 32 * self.n_workers
        self.epochs = epochs
        self.epsilon = epsilon
        self.lr = lr

        self.current_policy = Model(self.state_shape, self.n_actions).to(self.device)
        self.old_policy = Model(self.state_shape, self.n_actions).to(self.device)
        self.old_policy.load_state_dict(self.current_policy.state_dict())

        self.optimizer = Adam(self.current_policy.parameters(), lr=self.lr)

    def get_action_and_values(self, state):
        state = from_numpy(state).byte().permute([0, 3, 1, 2]).to(self.device)
        with torch.no_grad():
            dist, value = self.current_policy(state)
            action = dist.sample().cpu().numpy()
        return action, value.detach().cpu().numpy()

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            idxes = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[idxes], actions[idxes], returns[idxes], advs[idxes]

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_gae(rewards, values.copy(), next_values, dones)
        advs = returns - np.vstack(values).reshape((sum([len(values[i]) for i in range(self.n_workers)]), 1))
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for epoch in range(self.epochs):
            for state, action, q_value, adv in self.choose_mini_batch(self.mini_batch_size,
                                                                      states, actions, returns, advs):
                state = torch.ByteTensor(state).permute([0, 3, 1, 2]).to(self.device)
                action = torch.Tensor(action).to(self.device)
                adv = torch.Tensor(adv).to(self.device)
                q_value = torch.Tensor(q_value).to(self.device).view((self.mini_batch_size, 1))

                dist, value = self.current_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.current_policy, state, action)
                with torch.no_grad():
                    old_log_prob = self.calculate_log_probs(self.old_policy, state, action)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_ac_loss(ratio, adv)
                critic_loss = (q_value - value).pow(2).mean()

                total_loss = 1.0 * critic_loss + actor_loss - 0.01 * entropy
                self.equalize_policies()
                self.optimize(total_loss)

                return total_loss, entropy, rewards

    def equalize_policies(self):
        for old_params, new_params in zip(self.old_policy.parameters(), self.current_policy.parameters()):
            old_params.data.copy_(new_params.data)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(), 0.5)
        self.optimizer.step()

    def get_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):

        returns = [[] for _ in range(self.n_workers)]
        extended_values = np.zeros((self.n_workers, len(rewards[0]) + 1))
        for worker in range(self.n_workers):
            extended_values[worker] = np.append(values[worker], next_values[worker])
            gae = 0
            for step in reversed(range(len(rewards[worker]))):
                delta = rewards[worker][step] + gamma * (extended_values[worker][step + 1]) * (1 - dones[worker][step]) - \
                        extended_values[worker][step]
                gae = delta + gamma * lam * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + extended_values[worker][step])

        return np.vstack(returns).reshape((sum([len(returns[i]) for i in range(self.n_workers)]), 1))

    def calculate_ratio(self, states, actions):
        new_policy_log = self.calculate_log_probs(self.current_policy, states, actions)
        old_policy_log = self.calculate_log_probs(self.old_policy, states, actions)
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