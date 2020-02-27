import gym
from agent import Agent
from train import Train

ENV_NAME = "Breakout-v0"
test_env = gym.make(ENV_NAME)

state_shape = test_env.observation_space.shape
n_actions = test_env.action_space.n

stack_shape = (84, 84, 4)
n_workers = 3
max_steps = 256#test_env._max_episode_steps
max_episode = 500
lr = 3e-4
epochs = 4
clip_range=0.2

if __name__ == "__main__":
    print(f"state shape:{state_shape}\n"
          f"number of actions:{n_actions}")
    env = gym.make(ENV_NAME)
    agent = Agent(state_shape=stack_shape, n_actions=n_actions, lr=lr)
    trainer = Train(env=env,
                    agent=agent,
                    n_workers=n_workers,
                    max_steps=max_steps,
                    max_episode=max_episode,
                    epochs=epochs,
                    mini_batch_size=32 * n_workers,
                    epsilon=clip_range
                    )
    trainer.step()

#Todo
#correct gae horizon
