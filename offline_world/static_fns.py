"""
Reference: https://github.com/tianheyu927/mopo/tree/master/mopo/static
    where MOPO and RAMBO use the original gym termination conditions.
Reference: https://github.com/yihaosun1124/mobile/blob/main/utils/termination_fns.py
    where MOBILE and MAPLE adds np.all(np.abs(next_obs) < 100, axis=-1)
    to walker2d or halfcheetah tasks, which we do not adopt. But this change might be minor.
Reference: https://github.com/kwanyoungpark/LEQ/blob/main/dynamics/termination_fns.py
    where LEQ uses a radius of 0.25 for antmaze tasks, which we also adopt.
"""

import gym, d4rl
import numpy as np


def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros(len(obs), dtype=bool)
    return done


def termination_fn_neorl_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros(len(obs), dtype=bool)
    return done


def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (
        np.isfinite(next_obs).all(axis=-1)
        * (np.abs(next_obs[:, 1:]) < 100).all(axis=-1)  # fixed this abs bug
        * (height > 0.7)
        * (np.abs(angle) < 0.2)
    )

    done = ~not_done
    return done


def termination_fn_neorl_hopper(obs, act, next_obs):
    # same fn as d4rl hopper, but take the extra dim in neorl into consideration
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 1]
    angle = next_obs[:, 2]
    state = next_obs[:, 2:]

    min_state, max_state = (-100.0, 100.0)
    min_z, max_z = (0.7, float("inf"))
    min_angle, max_angle = (-0.2, 0.2)

    healthy_state = np.all(
        np.logical_and(min_state < state, state < max_state), axis=-1
    )
    healthy_z = np.logical_and(min_z < z, z < max_z)
    healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

    is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)

    done = np.logical_not(is_healthy)
    return done


def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    done = ~not_done
    return done


def termination_fn_neorl_walker2d(obs, act, next_obs):
    # same fn as d4rl hopper, but take the extra dim in neorl into consideration
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 1]
    angle = next_obs[:, 2]

    min_z, max_z = (0.8, 2.0)
    min_angle, max_angle = (-1.0, 1.0)

    healthy_z = np.logical_and(min_z < z, z < max_z)
    healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)
    is_healthy = np.logical_and(healthy_z, healthy_angle)
    done = np.logical_not(is_healthy)
    return done


def termination_fn_antmaze(next_obs, goal_pos, radius):
    """
    NOTE: the goal_pos is the average goal position.
    The actual goal (not observed) follows a distribution of
        goal_cell + uniform[0.0, 1.0] + uniform[0.0, 0.5]
    """
    goal_pos = np.tile(goal_pos, (next_obs.shape[0], 1))
    dist = np.linalg.norm(next_obs[:, :2] - goal_pos, axis=-1)
    done = dist < radius
    return done


def termination_fn_antmaze_umaze(obs, act, next_obs):
    goal = (0.75, 8.75)
    radius = 0.25
    return termination_fn_antmaze(next_obs, goal, radius)


def termination_fn_antmaze_medium(obs, act, next_obs):
    goal = (20.75, 20.75)
    radius = 0.25
    return termination_fn_antmaze(next_obs, goal, radius)


def termination_fn_antmaze_large(obs, act, next_obs):
    goal = (32.75, 24.75)
    radius = 0.25
    return termination_fn_antmaze(next_obs, goal, radius)


def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    obj_pos = next_obs[:, 24:27]
    done = obj_pos[:, 2] < 0.075
    return done


def termination_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros(len(obs), dtype=bool)
    return done


def termination_fn_hammer(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros(len(obs), dtype=bool)
    return done


def get_termination_fn(task: str):
    if "halfcheetah-" in task:
        return termination_fn_halfcheetah
    elif "hopper-" in task:
        return termination_fn_hopper
    elif "walker2d-" in task:
        return termination_fn_walker2d
    elif "antmaze-umaze" in task:
        return termination_fn_antmaze_umaze
    elif "antmaze-medium" in task:
        return termination_fn_antmaze_medium
    elif "antmaze-large" in task:
        return termination_fn_antmaze_large
    elif "pen-" in task:
        return termination_fn_pen
    elif "door-" in task:
        return termination_fn_door
    elif "hammer-" in task:
        return termination_fn_hammer
    elif "HalfCheetah-v3" in task:
        return termination_fn_neorl_halfcheetah
    elif "Hopper-v3" in task:
        return termination_fn_neorl_hopper
    elif "Walker2d-v3" in task:
        return termination_fn_neorl_walker2d
    else:
        raise NotImplementedError


"""
Only used in debugging on model errors; 
not used in training or evaluating the agent performance
"""


def extract_qpos_qvel_halfcheetah(obs):
    # qpos starts from the second element of observation (17 elements in total)
    # The first 8 elements of qpos
    qpos = np.concatenate([[0], obs[:8]])  # Adding 0 for the missing first qpos element
    # qvel are the last 9 elements of observation
    qvel = obs[8:17]
    return qpos, qvel


def extract_qpos_qvel_hopper(obs):
    # Hopper observation typically starts with qpos (excluding the first element) followed by qvel
    qpos_dim = 5  # number of qpos elements (excluding the first element)
    qvel_dim = 6  # number of qvel elements
    qpos = np.concatenate(
        [[0], obs[:qpos_dim]]
    )  # Adding 0 for the missing first qpos element
    qvel = obs[qpos_dim : qpos_dim + qvel_dim]

    return qpos, qvel


def extract_qpos_qvel_walker2d(obs):
    # Walker2d observation typically starts with qpos (excluding the first element) followed by qvel
    qpos_dim = 8  # number of qpos elements (excluding the first element)
    qvel_dim = 9  # number of qvel elements
    qpos = np.concatenate(
        [[0], obs[:qpos_dim]]
    )  # Adding 0 for the missing first qpos element
    qvel = obs[qpos_dim : qpos_dim + qvel_dim]

    return qpos, qvel


class OracleEnv:
    def __init__(self, task):
        if "halfcheetah-" in task:
            self.extract_qpos_qvel = extract_qpos_qvel_halfcheetah
        elif "hopper-" in task:
            self.extract_qpos_qvel = extract_qpos_qvel_hopper
        elif "walker2d-" in task:
            self.extract_qpos_qvel = extract_qpos_qvel_walker2d
        else:
            print("OracleEnv not implemented")
            return
        self.env = gym.make(task)

    def reset(self, initial_state):
        # arbitrary initial state at any timestep
        qpos, qvel = self.extract_qpos_qvel(initial_state)
        self.env.reset()
        self.env.set_state(qpos, qvel)

    def step(self, action):
        next_state, reward, _, _ = self.env.step(action)
        return next_state, reward
