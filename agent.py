from model import Model
from torch.optim import Adam
from torch import from_numpy
from torch.distributions import Categorical
from torch.nn import MSELoss
import numpy as np
import torch



class Agent:
    def __init__(self, state_shape, n_actions, lr):
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lr = lr
        self.new_policy = Model(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)
        self.old_policy = Model(state_shape=self.state_shape, n_actions=self.n_actions).to(self.device)

        self.old_policy.load_state_dict(self.new_policy.state_dict())

        self.optimizer = Adam(self.new_policy.parameters(), lr=self.lr)
        self.critic_loss = MSELoss()

    def choose_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float().permute([0, 3, 1, 2]).to(self.device)
        dist, v = self.new_policy(state)
        action = dist.sample().cpu()

        return action, v

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.new_policy.parameters(), 0.5)
        self.optimizer.step()

    def set_weights(self):
        for old_params, new_params in zip(self.old_policy.parameters(), self.new_policy.parameters()):
            old_params.data.copy_(new_params.data)

    def save_weights(self):
        # torch.save(self.actor.state_dict(), "./actor_weights.pth")
        # torch.save(self.critic.state_dict(), "./critic_weights.pth")
        torch.save(self.new_policy.state_dict(), "./weights.pth")

    def load_weights(self):
        # self.actor.load_state_dict(torch.load("./actor_weights.pth"))
        # self.critic.load_state_dict(torch.load("./critic_weights.pth"))
        pass

    def set_to_eval_mode(self):
        # self.actor.eval()
        # self.critic.eval()
        pass



