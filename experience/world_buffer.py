import numpy as np
import jax
import jax.numpy as jnp
from experience.evaluator import Transition
from dataclasses import replace
from tqdm import tqdm
import time

tuple_keys = Transition.__annotations__.keys()


def get_dataset(domain: str, env, **kwargs):
    if "neorl" in domain:
        import neorl

        # this setup follows prior work including MOBILE and LEQ
        train_data, _ = env.get_dataset(train_num=1000, need_val=False)
        dataset = {}
        dataset["observations"] = train_data["obs"]
        dataset["actions"] = train_data["action"]
        dataset["next_observations"] = train_data["next_obs"]
        dataset["rewards"] = train_data["reward"].squeeze()

        dones = train_data["done"].squeeze()  # timeout or terminal
        end_idx = np.flatnonzero(dones)
        start_idx = np.empty_like(end_idx)
        start_idx[0] = 0
        start_idx[1:] = end_idx[:-1] + 1
        assert np.all(start_idx == train_data["index"])
        lengths = end_idx - start_idx + 1

        dataset["terminals"] = np.zeros(dataset["rewards"].shape, dtype=bool)
        dataset["timeouts"] = np.zeros(dataset["rewards"].shape, dtype=bool)

        dataset["terminals"][end_idx] = lengths < env._max_episode_steps
        dataset["timeouts"][end_idx] = lengths >= env._max_episode_steps

        return dataset
    else:  # d4rl
        return env.get_dataset(**kwargs)


def world_learning_dataset(
    domain: str, env, dataset=None, terminate_on_end=False, **kwargs
):
    """
    https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/utils/load_dataset.py

    Returns datasets formatted for use by **world modeling**. This follows the D4RL's
        qlearning_dataset function, with minor changes to use next_obs if available.
    As a note, some D4RL tasks (Adroit, AntMaze) may have trajectories longer than
        the environment's max horizon, as indicated by timeouts. We still include
        full trajectories in this function for (1) better world model generalizablity
        (2) fair comparison with other model-based methods.
        However, we use truncated trajectories in agent training,
        see get_history_dataset for details.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of episode termination flags.
    """
    if dataset is None:
        dataset = get_dataset(domain, env, **kwargs)

    has_next_obs = True if "next_observations" in dataset.keys() else False
    assert "timeouts" in dataset

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []

    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        if has_next_obs:  # Change 1: use next_observations if available
            new_obs = dataset["next_observations"][i].astype(np.float32)
        else:
            new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        # use the timeouts field here, same as prior method
        final_timestep = bool(dataset["timeouts"][i])

        if (not terminate_on_end) and final_timestep:
            # Skip timeout transition
            continue
        if done_bool or final_timestep:
            if not has_next_obs:
                # Change 2: skip the last transition if next_obs is not available,
                # appropriate for world model training, but not q learning
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)

    world_data = {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
    }
    if "antmaze" in domain:
        assert world_data["rewards"].max() == 0.0 and world_data["rewards"].min() == 0.0
        # follow LEQ https://github.com/kwanyoungpark/LEQ/blob/main/train/train_MOBILE.py#L132
        world_data["rewards"] -= 1.0
    return world_data


def get_history_dataset(domain: str, env, dataset=None, **kwargs):
    """
    Return a list of trajectories in the Tape format.
    We truncate trajectories up to the max horizon for **agent training**,
        because (1) this aligns with testing environments,
        (2) our agent is history-dependent and thus sensitive to timestep.
    """
    if dataset is None:
        dataset = get_dataset(domain, env, **kwargs)

    has_next_obs = True if "next_observations" in dataset.keys() else False
    if has_next_obs:  # mujoco
        N = dataset["rewards"].shape[0]
        max_horizon = env._max_episode_steps
    else:  # Adroit or AntMaze
        N = dataset["rewards"].shape[0] - 1
        max_horizon = env._max_episode_steps - 1

    histories = []
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_reward = np.array([0.0], dtype=np.float32)

    ## get boundaries of trajectories
    assert "timeouts" in dataset
    if "antmaze" in domain:
        assert np.array_equal(np.unique(dataset["rewards"]), np.array([0.0, 1.0]))
        assert np.all((dataset["rewards"] == 1.0) == dataset["terminals"])
        dataset["rewards"] -= 1.0  # follow LEQ
        dones = dataset["timeouts"]
    else:
        dones = np.logical_or(dataset["terminals"], dataset["timeouts"])

    done_indices = np.where(dones == True)[0].tolist()
    if done_indices[-1] != dataset["rewards"].shape[0] - 1:
        done_indices.append(dataset["rewards"].shape[0] - 1)

    start_idx = 0
    for raw_end_idx in done_indices:
        traj = []
        ## NOTE: we do not use timeouts field only, as it may divide data into
        # trajectories longer than max horizon in Adroit or AntMaze.
        end_idx = min(raw_end_idx + 1, start_idx + max_horizon)
        end_idx = min(end_idx, N)

        for i in range(start_idx, end_idx):
            obs = dataset["observations"][i].astype(np.float32)
            if has_next_obs:
                new_obs = dataset["next_observations"][i].astype(np.float32)
            else:
                new_obs = dataset["observations"][i + 1].astype(np.float32)
            action = dataset["actions"][i].astype(np.float32)
            reward = float(dataset["rewards"][i])

            if i == end_idx - 1 and not bool(dataset["terminals"][i]):
                # final step must be either terminal or timeout
                final_timestep = True
            else:
                final_timestep = False

            transition = Transition(
                start=(len(traj) == 0),
                observation=(
                    np.concatenate([obs, null_action, null_reward])
                    if len(traj) == 0
                    else np.concatenate(
                        [obs, traj[-1].action, np.array([traj[-1].next_reward])]
                    )
                ),
                action=action,
                next_reward=reward,
                next_observation=np.concatenate([new_obs, action, np.array([reward])]),
                next_terminated=bool(dataset["terminals"][i]),
                next_truncated=final_timestep,
            )
            traj.append(transition)

            if "antmaze" in domain and bool(dataset["terminals"][i]):
                # to align with test envs
                break

        if len(traj) > 0:
            histories.append(traj)
        start_idx = raw_end_idx + 1

    ## convert data to the expected format
    rollouts = []
    for traj in histories:
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
            for key in tuple_keys
            if key != "next_observation"
        }
        rollouts.append(stacked)

    return rollouts


class HistorySampler:
    def __init__(self, domain: str, env):
        self.history_dataset = get_history_dataset(domain, env)
        lengths = [len(h["observation"]) for h in self.history_dataset]
        self.cum_lengths = np.cumsum(lengths)
        self.max_length = max(lengths)
        self.obs_dim = self.history_dataset[0]["observation"].shape[-1]

    def sample(self, start_from_s0: bool, size: int, key: jax.random.PRNGKey):
        """
        Sample a batch of histories from the history dataset.
            observation and start up to timestep t and the rest items up to t-1,
            where t = 0 to the end of the trajectory (included)
        """
        rng = np.random.default_rng(jax.random.bits(key).item())

        if start_from_s0:  # uniformly from t=0
            traj_idx = rng.integers(len(self.history_dataset), size=size)
            offsets = np.zeros(size, dtype=int)

        else:  # uniformly from any timestep
            global_idx = rng.integers(self.cum_lengths[-1], size=size)
            traj_idx = np.searchsorted(self.cum_lengths, global_idx, side="right")
            starts = np.concatenate(([0], self.cum_lengths[:-1]))
            offsets = global_idx - starts[traj_idx]

        histories = []
        for t_idx, o_idx in zip(traj_idx, offsets):
            history = {
                k: (
                    self.history_dataset[t_idx][k][: o_idx + 1]  # include t
                    if k in ["observation", "start"]
                    else self.history_dataset[t_idx][k][:o_idx]  # up to t-1
                )
                for k in tuple_keys
                if k != "next_observation"
            }

            histories.append(history)

        # a fast way to left-pad the observations and starts
        padded_obs = np.zeros(
            (size, self.max_length, self.obs_dim),
            dtype=np.float32,
        )
        padded_start = np.ones((size, self.max_length), dtype=bool)
        for i, history in enumerate(histories):
            padded_obs[i, -len(history["observation"]) :] = history["observation"]
            padded_start[i, -len(history["observation"]) :] = history["start"]

        infos = {
            "padded_obs": jnp.array(padded_obs),
            "padded_start": padded_start,
            # extract the elapsed timesteps, starting from 0
            "timesteps": np.array([len(h["observation"]) for h in histories]) - 1,
            # extract the last terminal flags
            "terminateds": np.array(
                [
                    h["next_terminated"][-1] if len(h["next_terminated"]) > 0 else False
                    for h in histories
                ]
            ),
            # extract the last truncated flags
            "truncateds": np.array(
                [
                    h["next_truncated"][-1] if len(h["next_truncated"]) > 0 else False
                    for h in histories
                ]
            ),
        }

        return histories, infos
