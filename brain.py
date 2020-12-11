from model import Model
import torch
from torch import from_numpy
import numpy as np
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils import explained_variance


class Brain:
    def __init__(self, state_shape, n_actions, device, n_workers, epochs, n_iters, epsilon, lr):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.n_workers = n_workers
        self.mini_batch_size = 32
        self.epochs = epochs
        self.n_iters = n_iters
        self.initial_epsilon = epsilon
        self.epsilon = self.initial_epsilon
        self.lr = lr

        self.current_policy = Model(self.state_shape, self.n_actions).to(self.device)

        self.optimizer = Adam(self.current_policy.parameters(), lr=self.lr, eps=1e-5)
        self._schedule_fn = lambda step: max(1.0 - float(step / self.n_iters), 0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._schedule_fn)

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).byte().to(self.device)
        with torch.no_grad():
            dist, value = self.current_policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), value.detach().cpu().numpy().squeeze(), log_prob.cpu().numpy()

    def choose_mini_batch(self, states, actions, returns, advs, values, log_probs):
        full_batch_size = len(states)
        states = torch.ByteTensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        advs = torch.Tensor(advs).to(self.device)
        returns = torch.Tensor(returns).to(self.device)
        values = torch.Tensor(values).to(self.device)
        log_probs = torch.Tensor(log_probs).to(self.device)

        indices = np.random.randint(0, full_batch_size, (self.n_workers, self.mini_batch_size))

        for idx in indices:
            yield states[idx], actions[idx], advs[idx], returns[idx], values[idx], \
                  log_probs[idx]

    def train(self, states, actions, rewards, dones, values, log_probs, next_values):
        returns = self.get_gae(rewards, values.copy(), next_values, dones)
        values = np.concatenate(values)
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # pg_losses, v_losses, entropies = [], [], []
        for epoch in range(self.epochs):
            for state, action, q_value, adv, old_value, old_log_prob in self.choose_mini_batch(states, actions, returns,
                                                                                               advs, values, log_probs):

                dist, value = self.current_policy(state)
                entropy = dist.entropy().mean()
                new_log_prob = self.calculate_log_probs(self.current_policy, state, action)
                ratio = (new_log_prob - old_log_prob).exp()
                actor_loss = self.compute_ac_loss(ratio, adv)

                clipped_value = old_value + torch.clamp(value.squeeze() - old_value, -self.epsilon, self.epsilon)
                clipped_v_loss = (clipped_value - q_value).pow(2)
                unclipped_v_loss = (value.squeeze() - q_value).pow(2)
                critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()

                total_loss = critic_loss + actor_loss - 0.01 * entropy
                self.optimize(total_loss)

                # pg_losses.append(actor_loss.item())
                # v_losses.append(critic_loss.item())
                # entropies.append(entropy.item())

        return total_loss.item(), entropy.item(), explained_variance(values, returns)

    def schedule_lr(self):
        self.scheduler.step()

    def schedule_clip_range(self, iter):
        self.epsilon = max(1.0 - float(iter / self.n_iters), 0) * self.initial_epsilon

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
                delta = rewards[worker][step] + \
                        gamma * (extended_values[worker][step + 1]) * (1 - dones[worker][step]) \
                        - extended_values[worker][step]
                gae = delta + gamma * lam * (1 - dones[worker][step]) * gae
                returns[worker].insert(0, gae + extended_values[worker][step])

        return np.concatenate(returns)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution, _ = model(states)
        return policy_distribution.log_prob(actions)

    def compute_ac_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean()
        return loss

    def save_params(self, iteration, running_reward, episode):
        torch.save({"current_policy_state_dict": self.current_policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "iteration": iteration,
                    "episode": episode,
                    "running_reward": running_reward,
                    "clip_range": self.epsilon},
                   "params.pth")

    def load_params(self):
        checkpoint = torch.load("params.pth", map_location=self.device)
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        episode = checkpoint["episode"]
        running_reward = checkpoint["running_reward"]
        self.epsilon = checkpoint["clip_range"]

        return running_reward, iteration, episode

    def set_to_eval_mode(self):
        self.current_policy.eval()
