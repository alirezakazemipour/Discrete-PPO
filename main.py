from runner import Worker
from torch.multiprocessing import Process, Pipe
import numpy as np
from numpy import asarray
from brain import Brain
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from play import Play

env_name = "SuperMarioBros-1-1-v0"
test_env = gym_super_mario_bros.make(env_name)
test_env = JoypadSpace(test_env, SIMPLE_MOVEMENT)
n_actions = test_env.action_space.n
n_workers = 8
state_shape = (4, 84, 84)
device = "cuda"
iterations = 24000
log_period = 10
T = 128
epochs = 3
lr = 2.5e-4
clip_range = 0.1
LOAD_FROM_CKP = True
Train = True


def run_workers(worker, conn):
    worker.step(conn)


def receive(p):
    return p.recv()


def send_action(x):
    x[0].send(int(x[1]))


if __name__ == '__main__':
    brain = Brain(state_shape, n_actions, device, n_workers, epochs, iterations, clip_range, lr)
    if Train:
        if LOAD_FROM_CKP:
            running_reward, init_iteration, episode = brain.load_params()
        else:
            init_iteration = 0
            running_reward = 0
            episode = 0

        workers = [Worker(i, state_shape, env_name) for i in range(n_workers)]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            parents.append(parent_conn)
            p.start()

        init_total_states = np.zeros((n_workers, T,) + state_shape)
        init_total_actions = np.zeros((n_workers, T))
        init_total_rewards = np.zeros((n_workers, T))
        init_total_dones = np.zeros((n_workers, T), dtype=np.bool)
        init_total_values = np.zeros((n_workers, T))
        init_total_log_probs = np.zeros((n_workers, T))
        init_next_states = np.zeros((n_workers,) + state_shape, dtype=np.uint8)
        init_next_values = np.zeros(n_workers)

        episode_reward = 0
        for iteration in tqdm(range(init_iteration + 1, iterations + 1)):
            start_time = time.time()
            total_states = init_total_states
            total_actions = init_total_actions
            total_rewards = init_total_rewards
            total_dones = init_total_dones
            total_values = init_total_values
            total_log_probs = init_total_log_probs
            next_states = init_next_states
            next_values = init_next_values

            for t in range(T):
                # for worker_id, parent in enumerate(parents):
                #     s = parent.recv()
                #     total_states[worker_id, t] = s
                total_states[:, t] = asarray(list(map(receive, parents)), dtype=np.uint8)

                total_actions[:, t], total_values[:, t], total_log_probs[:, t] = \
                    brain.get_actions_and_values(total_states[:, t], batch=True)
                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(int(a))
                # k = map(send_action, (parents, total_actions[:, t]))

                # for worker_id, parent in enumerate(parents):
                x = list(map(receive, parents))
                s_, r, d = asarray(x, dtype=object)[:, 0], asarray(x, dtype=object)[:, 1], asarray(x, dtype=object)[:, 2]
                total_rewards[:, t] = r
                total_dones[:, t] = d
                next_states[:] = np.concatenate(s_).reshape(n_workers, *state_shape)

                episode_reward += total_rewards[0, t]
                if total_dones[0, t]:
                    episode += 1
                    if episode == 1:
                        running_reward = episode_reward
                    else:
                        running_reward = 0.99 * running_reward + 0.01 * episode_reward
                    episode_reward = 0

            _, next_values, _ = brain.get_actions_and_values(next_states, batch=True)

            total_states = total_states.reshape((n_workers * T,) + state_shape)
            total_actions = total_actions.reshape(n_workers * T)
            total_log_probs = total_log_probs.reshape(n_workers * T)

            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            total_loss, entropy, ev = brain.train(total_states, total_actions, total_rewards,
                                                  total_dones, total_values, total_log_probs, next_values)
            brain.schedule_lr()
            brain.schedule_clip_range(iteration)

            if iteration % log_period == 0:
                print(f"Iter: {iteration}| "
                      f"Episode: {episode}| "
                      f"Ep_reward: {episode_reward:.1f}| "
                      f"Running_reward: {running_reward:.1f}| "
                      f"Total_loss: {total_loss:.3f}| "
                      f"Explained variance:{ev:.3f}| "
                      f"Entropy: {entropy:.3f}| "
                      f"Iter_duration: {time.time() - start_time:.3f}| "
                      f"Lr: {brain.scheduler.get_last_lr()}| "
                      f"Clip_range:{brain.epsilon:.3f}")
                brain.save_params(iteration, running_reward, episode)

            with SummaryWriter(env_name + "/logs") as writer:
                writer.add_scalar("running reward", running_reward, iteration)
                writer.add_scalar("episode reward", episode_reward, iteration)
                writer.add_scalar("explained variance", ev, iteration)
                writer.add_scalar("loss", total_loss, iteration)
                writer.add_scalar("entropy", entropy, iteration)
    else:
        play = Play(env_name, brain)
        play.evaluate()
