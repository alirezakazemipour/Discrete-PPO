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
        self.states = deque(maxlen=self.horizon)
        self.actions = deque(maxlen=self.horizon)
        self.rewards = deque(maxlen=self.horizon)
        self.dones = deque(maxlen=self.horizon)
        self.next_states = deque(maxlen=self.horizon)
        self.values = deque(maxlen=self.horizon)

    def __str__(self):
        return str(self.id)

    def render(self):
        # print(self.id)
        self.env.render()
        # time.sleep(1)

    def reset(self):
        self._state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, self._state, True)

    def step(self):
        action, value = self.brain.get_actions_and_values(self._stacked_states)
        # print(self.id, self.brain.epsilon)
        # self.render()
        next_state, r, d, _ = self.env.step(action)
        self.states.append(self._stacked_states)
        self.actions.append(action)
        self.rewards.append(r)
        self.dones.append(d)
        self.values.append(value)
        self._stacked_states = stack_states(self._stacked_states, next_state, False)
        self.next_states.append(self._stacked_states)
        if d:
            self.reset()
        # return dict({"next_state": next_state, "reward": r, "done": d})
        # for _ in range(3):
        #     conn.send(dict({"id": self.id, "next_state": next_state, "reward": r, "done": d}))
        # conn.close()
