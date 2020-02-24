import gym
from agent import Agent
from train import Train


ENV_NAME = "Breakout-v0"
test_env = gym.make(ENV_NAME)

state_shape = test_env.observation_space.shape
n_actions = test_env.action_space.n

n_workers = 1

print(f"state shape:{state_shape}\n"
      f"number of actions:{n_actions}")

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = Agent
    trainer = Train(env=env,
                    agent=agent,
                    n_workers=n_workers)
    trainer.train()