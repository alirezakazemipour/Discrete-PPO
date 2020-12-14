from Common.utils import *


class Worker:
    def __init__(self, id, **config):
        self.id = id
        self.config = config
        self.state_shape = self.config["state_shape"]
        self.env = make_mario(self.config["env_name"])
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self.score = 0
        self.reset()
        print(f"Worker {self.id}: initiated.")

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)
        self.score = 0

    def step(self, conn):
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, info = self.env.step(action)
            new_score = info["score"] - self.score
            self.score = info["score"]
            r = r + new_score / 40  # r + new_score -> would be clipped later.
            if d:
                if info["flag_get"]:
                    r += 350  # 50
                else:
                    r -= 50
            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            conn.send((self._stacked_states, r / 10, d, info))  # np.sign(r)
            if d:
                self.reset()
