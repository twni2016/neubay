"""
Generate offline datasets for the bandit problem
"""

import gym
from gym.wrappers import TimeLimit
import numpy as np
import jax
import jax.numpy as jnp


class Bandit(gym.Env):
    """
    A simple Bernoulli multi-armed bandit environment.
    p_list: list of probabilities of success (+1) for each arm
        R(a) = 1 with prob p_list[a] and 0 with prob 1 - p_list[a]
    """

    def __init__(self, p_list: list) -> None:
        self.p_list = p_list
        assert len(self.p_list) >= 2, "BanditEnv requires at least 2 arms"
        self.n_arms = len(self.p_list)

        self.action_space = gym.spaces.Discrete(self.n_arms)  # {0, 1, ..., n_arms-1}
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,))  # dummy

        self.reward_choices = [1.0, 0.0]  # discrete rewards

    def _get_obs(self):
        return np.array([0.0])  # dummy observation

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        reward = np.random.choice(
            self.reward_choices, p=[self.p_list[action], 1.0 - self.p_list[action]]
        )

        return self._get_obs(), reward, False, {"p_list": self.p_list}


def make_bandit(p_list):
    return TimeLimit(Bandit(p_list), max_episode_steps=100)


class VectorNaiveAgent:

    def initial_state(self, *args, **kwargs):
        return None

    def __call__(self, x, state, start, key):
        # always select the first arm
        return jnp.zeros((x.shape[0]), dtype=jnp.int32), None, None


if __name__ == "__main__":
    from experience.wrapper import ActionRewardWrapper
    from experience.evaluator import BanditEvaluator
    import os, time
    import pickle

    n_envs = 10
    env_fns = [
        lambda: ActionRewardWrapper(make_bandit([0.5, 0.5])) for _ in range(n_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)
    # Note that vectorization is only faster in simulation **when n_envs > 20**,
    #  a sweet spot is 100 - 1000 envs.
    #  Does not count for the agent inference time.

    evaluator = BanditEvaluator(envs)
    agent = VectorNaiveAgent()
    t0 = time.time()

    key = jax.random.PRNGKey(0)
    key, collect_key = jax.random.split(key)
    data, metrics = evaluator(agent, key=collect_key)

    print("ep_rewards", np.mean(metrics["reward"]), metrics["reward"])
    print("ep_lens", metrics["length"])
    print("time", time.time() - t0)

    #### store the dataset
    # parent_dir = "offline_world/data"
    # os.makedirs(parent_dir, exist_ok=True)
    # with open(f"{parent_dir}/bandit.pkl", "wb") as f:
    #     pickle.dump(data, f)
    # print(f"Dataset saved to {parent_dir}/bandit.pkl")

    # import pickle; data = pickle.load(open("offline_world/data/bandit.pkl", 'rb'))
