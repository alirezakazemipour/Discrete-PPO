from utils import *
from collections import deque


class Worker:
    def __init__(self, id, state_shape, env_name, brain, horizon):
        self.id = id
        self.env_name = env_name
        self.state_shape = state_shape
        self.brain = brain
        self.horizon = horizon
        self.env = make_atari(self.env_name)
        # self.env = gym.make(self.env_name)
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self._state = None
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        self._state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, self._state, True)

    def step(self, conn):
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, _ = self.env.step(action)
            # self.render()
            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            if d:
                self.reset()
            conn.send((self._stacked_states, r, d, _))
