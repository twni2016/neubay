import numpy as np
import gym, d4rl, neorl
from gym import spaces
from gym.wrappers import TimeLimit
from gym.core import ActType, ObsType
from typing import Tuple


def make_env(domain: str, dataset_name: str):
    if "neorl" in domain:
        env = neorl.make(dataset_name)
    elif "antmaze" in domain:
        env = AntMazeWrapper(gym.make(dataset_name))
    else:
        env = gym.make(dataset_name)
    return env


class AntMazeWrapper(gym.Wrapper):
    """
    Shift reward by -1 in antmaze following LEQ and IQL papers.
    As our policy is reward-conditioned, we need to have a wrapper on it
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert "antmaze" in env.spec.id
        self._max_episode_steps = env._max_episode_steps

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        reward -= 1.0
        return obs, reward, done, info

    def get_normalized_score(self, score):
        # add back the shift before calling this function
        return self.env.get_normalized_score(score)


def find_time_limit_wrapper(env: gym.Env):
    """
    Walk down .env links until we hit a TimeLimit wrapper.
    Returns the wrapper or None if it does not exist.
    """
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, TimeLimit):
            return current
        current = current.env  # move one level deeper
    return None


class MarkovWrapper(gym.Wrapper):
    """Used in ablation study for Markov agent."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    @property
    def max_episode_steps(self):
        """
        Returns the max episode step if a TimeLimit wrapper is present,
        otherwise None. Usually this wrapper is the last one in the chain.
        """
        tl = find_time_limit_wrapper(self)
        return None if tl is None else tl._max_episode_steps  # private attr


class ActionRewardWrapper(gym.Wrapper):
    """
    Appends the *previous action* (one‑hot if Discrete, raw if Box) and the
    *current reward* to every observation.

    Resulting observation: concat([obs_flat, action_vec, reward_scalar])
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Box)
        obs_low = env.observation_space.low.flatten().astype(np.float32)
        obs_high = env.observation_space.high.flatten().astype(np.float32)

        if isinstance(env.action_space, spaces.Discrete):
            # one-hot float encoding
            self._encode = lambda a: np.eye(env.action_space.n, dtype=np.float32)[
                int(a)
            ]
            act_low = np.zeros(env.action_space.n, dtype=np.float32)
            act_high = np.ones(env.action_space.n, dtype=np.float32)
            null_action = 0
        elif isinstance(env.action_space, spaces.Box):
            self._encode = lambda a: np.asarray(a, dtype=np.float32).flatten()
            act_low = env.action_space.low.flatten().astype(np.float32)
            act_high = env.action_space.high.flatten().astype(np.float32)
            null_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        else:
            raise NotImplementedError(
                "Only Discrete and Box action spaces are supported"
            )

        self.null_action_vec = self._encode(null_action)

        # ----- augmented observation space -----
        low = np.concatenate([obs_low, act_low, np.array([-np.inf], dtype=np.float32)])
        high = np.concatenate(
            [obs_high, act_high, np.array([np.inf], dtype=np.float32)]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def max_episode_steps(self):
        """
        Returns the max episode step if a TimeLimit wrapper is present,
        otherwise None. Usually this wrapper is the last one in the chain.
        """
        tl = find_time_limit_wrapper(self)
        return None if tl is None else tl._max_episode_steps  # private attr

    def reset(self, **kwargs) -> ObsType:
        obs = self.env.reset(**kwargs)
        augmented = np.concatenate(
            [
                obs.flatten().astype(np.float32),
                self.null_action_vec,
                np.array([0.0], dtype=np.float32),
            ]
        )
        return augmented

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        augmented = np.concatenate(
            [
                obs.flatten().astype(np.float32),
                self._encode(action),
                np.array([reward], dtype=np.float32),
            ]
        )
        return augmented, reward, done, info
