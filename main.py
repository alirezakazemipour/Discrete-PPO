from runner import Worker
from multiprocessing import Pool
import os
import numpy as np
from brain import Brain
import gym

env_name = "PongNoFrameskip-v4"
test_env = gym.make(env_name)
n_actions = test_env.action_space.n
test_env.close()
n_workers = 4
state_shape = (84, 84, 4)
device = "cuda"


def get_states(worker):
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())
    return worker.state


def apply_actions(worker, action):
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())
    return worker.step(action)


if __name__ == '__main__':
    workers = [Worker(i, env_name, state_shape) for i in range(n_workers)]
    brain = Brain(state_shape, n_actions, device)

    states = []
    rewards = []
    dones = []
    next_states = []
    with Pool(n_workers) as p:
        results = p.map(get_states, workers)
        for result in results:
            states.append(result)
    states = np.vstack(states).reshape((n_workers,) + state_shape)
    actions = brain.choose_action(states)

    with Pool(n_workers) as p:
        results = p.starmap(apply_actions, zip(workers, actions))
        for result in results:
            next_states.append(result["next_state"])
            rewards.append(result["reward"])
            dones.append(result["done"])
    next_states = np.vstack(next_states).reshape((n_workers,) + state_shape)
    rewards = np.vstack(rewards).reshape((n_workers, -1))
    dones = np.vstack(dones).reshape((n_workers, -1))

