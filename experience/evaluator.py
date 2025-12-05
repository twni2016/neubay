import numpy as np
from jax import random
import jax.numpy as jnp
from dataclasses import dataclass, replace
from online_rl.common import feature_metrics
import gym, d4rl
import time


class BanditEvaluator:
    """
    This is a simplified version of the evaluator, designed for fixed-horizon bandit problems.

    Evaluating the agent in a vectorized **real** environment.
    """

    def __init__(self, envs):
        # vectorized envs
        self.envs = envs

    def __call__(self, agent, key: random.PRNGKey):
        starts = []  # (s0, s1, ..., sT) where s0 = True, the rest = False
        observations = []  # (o0, o1, ..., oT) where oT is last observation
        actions = []  # (a0, a1, ..., aT) where aT is dummy action
        next_rewards = []  # (r1, r2, ..., rT+1) where rT+1 is dummy reward
        terminateds = []  # (d1, d2, ..., dT+1) where dT+1 is dummy terminated
        truncateds = []  # (d1, d2, ..., dT+1) where dT+1 is dummy terminated

        key, reset_key = random.split(key)
        observation = self.envs.reset(seed=random.bits(reset_key).item())
        done = False
        start = np.array([True] * self.envs.num_envs)

        starts.append(start)
        observations.append(observation)  # (N, dim)

        # get initial states: L[(N, 1, d_hidden)] complex
        recurrent_state = agent.initial_state(self.envs.num_envs)
        features = []  # report the plasticity metrics on features

        while not done:
            key, action_key = random.split(key)
            action, recurrent_state, feature = agent(
                x=jnp.expand_dims(observation, 1),  # (N, 1, dim)
                state=recurrent_state,
                start=jnp.expand_dims(start, 1),  # (N, 1)
                key=action_key,
            )
            action = np.array(action)  # (N)
            (
                observation,
                reward,
                dones,
                _,
            ) = self.envs.step(action)

            start = np.array([False] * self.envs.num_envs)
            starts.append(start)
            observations.append(observation)

            actions.append(action)
            next_rewards.append(reward)

            terminated = np.array([False] * self.envs.num_envs)
            truncated = dones  # assume truncation only
            terminateds.append(terminated)
            truncateds.append(truncated)

            if feature is not None:  # neural-based agent
                features.append(feature)

            done = dones.all()

        running_reward = np.array(next_rewards).sum(axis=0)
        step = len(next_rewards)
        metrics = {"reward": running_reward, "length": step}

        if features:
            features = np.concatenate(features, axis=0)  # (*, dim)
            metrics.update(feature_metrics(features))

        # Add dummy action, reward, terminated, truncated (will be ignored)
        actions.append(action)
        next_rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)

        data = {
            "start": np.array(starts),
            "observation": np.array(observations),
            "action": np.array(actions),
            "next_reward": np.array(next_rewards),
            "next_terminated": np.array(terminateds),
            "next_truncated": np.array(truncateds),
        }  # all with shape (T+1, N, *)

        # convert to list of dicts
        transitions = [
            {k: v[:, n] for k, v in data.items()} for n in range(self.envs.num_envs)
        ]

        return transitions, metrics


@dataclass
class Transition:
    start: bool
    observation: np.ndarray
    action: np.ndarray
    next_reward: float
    next_observation: np.ndarray
    next_terminated: bool
    next_truncated: bool


class ContEvaluator:
    """
    Evaluating the agent in a vectorized **real** continuous environment.
    """

    def __init__(self, envs):
        # vectorized envs
        self.envs = envs
        self.max_horizon = self.envs.envs[0].max_episode_steps
        self.get_normalized_score = self.envs.envs[0].get_normalized_score

    def __call__(self, agent, deterministic: bool, key: random.PRNGKey):
        data = []
        buffer = [[] for _ in range(self.envs.num_envs)]
        # buffer[n] is a running list of transitions in the n-th environment: each transition is a tuple
        #   (start, observation, action, next_reward, next_observation, terminated, truncated).
        # Once the trajectory is finished, it will be pushed to data, and buffer[n] is reset to empty list

        key, reset_key = random.split(key)
        reset_seed = random.bits(reset_key).item()
        try:
            observation = self.envs.reset(seed=reset_seed)
        except TypeError:  # Adroit
            self.envs.seed(seed=reset_seed)
            observation = self.envs.reset()

        start = np.array([True] * self.envs.num_envs)

        # get initial states: L[(N, 1, d_hidden)] complex
        recurrent_state = agent.initial_state(self.envs.num_envs)
        features = []  # report the plasticity metrics on features

        for t in range(self.max_horizon):
            key, action_key = random.split(key)
            action, recurrent_state, feature = agent(
                x=jnp.expand_dims(observation, 1),  # (N, 1, dim)
                state=recurrent_state,
                start=jnp.expand_dims(start, 1),  # (N, 1)
                key=action_key,
                deterministic=deterministic,
            )

            action = np.array(action)  # (N, dim)
            (
                next_observation,
                reward,
                done,
                info,
            ) = self.envs.step(action)

            if feature is not None:  # neural-based agent
                features.append(feature)

            for n in range(self.envs.num_envs):
                # collect transitions for each environment
                if done[n]:
                    if (
                        "TimeLimit.truncated" in info[n]
                        and info[n]["TimeLimit.truncated"]
                    ):
                        term = False
                        trunc = True
                    else:
                        term = True
                        trunc = False
                else:
                    term = trunc = False

                transition = Transition(
                    start=start[n],
                    observation=observation[n],
                    action=action[n],
                    next_reward=reward[n],
                    next_observation=(
                        next_observation[n]
                        if not done[n]
                        else info[n]["terminal_observation"]
                    ),
                    next_terminated=term,
                    next_truncated=trunc,
                )
                buffer[n].append(transition)
                if done[n]:
                    # trajectory finished, push to data
                    data.append(buffer[n])
                    buffer[n] = []

            observation = next_observation
            start = done.copy()  # reset the start flag for finished environments

        # stats for complete trajectories
        if "antmaze" in self.envs.envs[0].spec.id:
            # as we subtracted 1 in reward, we need to add it back
            returns = [sum(t.next_reward + 1.0 for t in traj) for traj in data]
        else:
            returns = [sum(t.next_reward for t in traj) for traj in data]

        lengths = [len(traj) for traj in data]
        metrics = {
            "reward": np.mean(returns),
            "reward_std": np.std(returns),
            "length": np.mean(lengths),
            "length_std": np.std(lengths),
        }
        metrics["normalized_score"] = self.get_normalized_score(metrics["reward"])

        if features:
            features = np.concatenate(features, axis=0)  # (*, dim)
            metrics.update(feature_metrics(features))

        # push the unfinished (truncated) trajectories in the buffer to data
        for traj in buffer:
            if len(traj) > 0:
                traj[-1] = replace(traj[-1], next_truncated=True)
                data.append(traj)

        total_steps = sum(len(traj) for traj in data)
        metrics["total_steps"] = total_steps

        # finally convert data to the expected format
        rollouts = []
        for traj in data:
            # add synthetic terminal transition
            terminal = replace(
                traj[-1],
                start=False,
                observation=traj[-1].next_observation,
                # dummy action, reward, terminated, truncated (will be ignored)
            )
            traj_ext = traj + [terminal]

            stacked = {
                key: np.stack([getattr(t, key) for t in traj_ext])
                for key in [
                    "start",
                    "observation",
                    "action",
                    "next_reward",
                    "next_terminated",
                    "next_truncated",
                ]
            }
            rollouts.append(stacked)

        return rollouts, metrics

    def call_markov(self, agent, deterministic: bool, key: random.PRNGKey):
        """
        __call__ for Markov agents
        """
        data = []
        buffer = [[] for _ in range(self.envs.num_envs)]
        # buffer[n] is a running list of transitions in the n-th environment: each transition is a tuple
        #   (observation, action, next_reward, next_observation, terminated, truncated).
        # Once the trajectory is finished, it will be pushed to data, and buffer[n] is reset to empty list

        key, reset_key = random.split(key)
        reset_seed = random.bits(reset_key).item()
        try:
            observation = self.envs.reset(seed=reset_seed)
        except TypeError:  # Adroit
            self.envs.seed(seed=reset_seed)
            observation = self.envs.reset()

        for t in range(self.max_horizon):
            key, action_key = random.split(key)
            action = agent(
                x=observation,  # (N, dim)
                key=action_key,
                eval_mode=True,
                deterministic=deterministic,
            )

            action = np.array(action)  # (N, dim)
            (
                next_observation,
                reward,
                done,
                info,
            ) = self.envs.step(action)

            for n in range(self.envs.num_envs):
                # collect transitions for each environment
                if done[n]:
                    if (
                        "TimeLimit.truncated" in info[n]
                        and info[n]["TimeLimit.truncated"]
                    ):
                        term = False
                        trunc = True
                    else:
                        term = True
                        trunc = False
                else:
                    term = trunc = False

                transition = Transition(
                    start=None,
                    observation=observation[n],
                    action=action[n],
                    next_reward=reward[n],
                    next_observation=(
                        next_observation[n]
                        if not done[n]
                        else info[n]["terminal_observation"]
                    ),
                    next_terminated=term,
                    next_truncated=trunc,
                )
                buffer[n].append(transition)
                if done[n]:
                    # trajectory finished, push to data
                    data.append(buffer[n])
                    buffer[n] = []

            observation = next_observation

        # stats for complete trajectories
        if "antmaze" in self.envs.envs[0].spec.id:
            # as we subtracted 1 in reward, we need to add it back
            returns = [sum(t.next_reward + 1.0 for t in traj) for traj in data]
        else:
            returns = [sum(t.next_reward for t in traj) for traj in data]

        lengths = [len(traj) for traj in data]
        metrics = {
            "reward": np.mean(returns),
            "reward_std": np.std(returns),
            "length": np.mean(lengths),
            "length_std": np.std(lengths),
        }
        metrics["normalized_score"] = self.get_normalized_score(metrics["reward"])

        # push the unfinished (truncated) trajectories in the buffer to data
        for traj in buffer:
            if len(traj) > 0:
                traj[-1] = replace(traj[-1], next_truncated=True)
                data.append(traj)

        total_steps = sum(len(traj) for traj in data)
        metrics["total_steps"] = total_steps

        # finally convert data to the expected format
        rollouts = []
        for traj in data:
            stacked = {
                key: np.stack([getattr(t, key) for t in traj])
                for key in [
                    "observation",
                    "action",
                    "next_reward",
                    "next_observation",
                    "next_terminated",
                    "next_truncated",
                ]
            }
            rollouts.append(stacked)

        return rollouts, metrics
