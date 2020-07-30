from utils import *


class Worker:
    def __init__(self, n, env_name, state_shape):
        self.n = n
        self.env_name = env_name
        self.state_shape = state_shape
        self.env = make_atari(self.env_name)
        # self.env = gym.make("MontezumaRevengeNoFrameskip-v4")
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self._state = None
        self._ep_r = None
        self.reset()

    def __str__(self):
        return str(self.n)

    @property
    def state(self):
        return self._stacked_states

    def render(self):
        self.env.render()

    def reset(self):
        self._state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, self._state, True)
        self._ep_r = 0

    def step(self, action):
        next_state, r, d, _ = self.env.step(action)
        self.render()
        self._stacked_states = stack_states(self._stacked_states, next_state, False)
        self._ep_r += r
        stacked_states_copy = self._stacked_states.copy()
        if d:
            self.reset()
            self._ep_r = 0
        return dict({"next_state": stacked_states_copy, "reward": r, "done": d})
