from model import Model
import torch
from torch import from_numpy


class Brain:
    def __init__(self, state_shape, n_actions, device):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device

        self.current_policy = Model(self.state_shape, self.n_actions).to(self.device)

    def choose_action(self, state):
        state = from_numpy(state).byte().permute([0, 3, 1, 2]).to(self.device)
        with torch.no_grad():
            dist, _ = self.current_policy(state)
            action = dist.sample().cpu().numpy()
        return action
