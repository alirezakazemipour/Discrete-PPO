import numpy as np
import cv2
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocessing(x):
    img = rgb2gray(x)  # / 255.0 -> Do it later in order to open up more RAM !!!!
    img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    img = img[18:102, :]
    return img


def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def make_mario(env_id):
    main_env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(main_env, SIMPLE_MOVEMENT)
    assert 'SuperMarioBros' in main_env.spec.id
    env = RepeatActionEnv(env)
    # env = EpisodicLifeEnv(env)
    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = 30
        self.noop_action = 0
        self.env = env
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'
        self.observation_space = self.env.observation_space

    def reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0

        state = None
        for _ in range(noops):
            state, _, done, _ = self.env.step(self.noop_action)
            if done:
                state = self.env.reset()

        return state

    def step(self, action):
        return self.env.step(action)


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        super(RepeatActionEnv, self).__init__(env)
        self.env = env
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.env = env
        self.natural_done = True
        self.lives = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.natural_done = done

        if self.lives > info["life"] > 0:
            done = True
        self.lives = info["life"]

        return state, reward, done, info

    def reset(self):
        if self.natural_done:
            state = self.env.reset()
            self.lives = 2  # self.env.smb_env._life
        else:
            state, _, _, info = self.env.step(0)
            self.lives = info["life"]
        return state
