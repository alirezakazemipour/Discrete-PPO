from utils import *


class Worker:
    def __init__(self, id, state_shape, env_name):
        self.id = id
        self.env_name = env_name
        self.state_shape = state_shape
        self.env = make_mario(self.env_name)
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self.score = 0
        self.reset()
        print(f"Worker: {self.id} initiated.")

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)

    def step(self, conn):
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, info = self.env.step(action)
            new_score = info["score"] - self.score
            self.score = info["score"]
            r = r + new_score + int(info["flag_get"])
            # if d and not info["flag_get"]:
            #     r = -1
            # print(np.sign(r))
            # self.render()
            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            conn.send((self._stacked_states, r, d))
            if d:
                self.score = 0
                self.reset()
