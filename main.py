from runner import Worker
from concurrent import futures
import cv2
import numpy as np
from brain import Brain
import gym
from tqdm import tqdm

env_name = "PongNoFrameskip-v4"
test_env = gym.make(env_name)
n_actions = test_env.action_space.n
n_workers = 8
state_shape = (84, 84, 4)
device = "cuda"
iterations = 1000
T = 128
epochs = 3
lr = 2.5e-4
clip_range = 0.1


def get_states(worker):
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())
    return worker.stacked_states()


def render(state, id):
    print("hello")
    cv2.imshow(f"obs:{id}", state)
    cv2.waitKey(1000)


def apply_actions(worker, action):
    return worker.step(action)


if __name__ == '__main__':
    workers = [Worker(i, env_name, state_shape) for i in range(n_workers)]
    brain = Brain(state_shape, n_actions, device, n_workers, epochs, clip_range, lr)
    for iteration in range(iterations):
        total_states = []
        total_actions = []
        total_rewards = []
        total_dones = []
        total_next_states = []
        total_values = []
        for step in tqdm(range(T)):
            states = []
            rewards = []
            dones = []
            next_states = []
            values = []
            with futures.ThreadPoolExecutor(n_workers) as p:
                results = p.map(get_states, workers)
                for result in results:
                    states.append(result)
                states = np.vstack(states).reshape((n_workers,) + state_shape)
                actions, values = brain.get_action_and_values(states)

                results = p.map(apply_actions, workers, actions)
                for result in results:
                    next_states.append(result["next_state"])
                    rewards.append(result["reward"])
                    dones.append(result["done"])

            # with futures.ProcessPoolExecutor(n_workers) as p:
            #     p.map(render, states, np.arange(n_workers))

            next_states = np.vstack(next_states).reshape((n_workers,) + state_shape)
            rewards = np.vstack(rewards).reshape((n_workers, -1))
            dones = np.vstack(dones).reshape((n_workers, -1))
            total_states.append(states)
            total_actions.append(actions)
            total_dones.append(dones)
            total_rewards.append(rewards)
            total_values.append(values)

        _, next_values = brain.get_action_and_values(next_states)
        next_values *= (1 - dones)
        total_states = np.vstack(total_states).reshape((n_workers * T,) + state_shape)
        total_actions = np.vstack(total_actions).reshape((n_workers * T, -1))
        total_values = np.vstack(total_values).reshape((n_workers, -1))
        total_rewards = np.vstack(total_rewards).reshape((n_workers, -1))
        total_dones = np.vstack(total_dones).reshape((n_workers, -1))
        brain.train(total_states, total_actions, total_rewards, total_dones, total_values, next_values)
