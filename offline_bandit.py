from jax import numpy as jnp
from jax import random
import jax
import numpy as np
import math
import time, os
from datetime import datetime
import equinox as eqx
import optax
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import gym
from offline_world.bandit_data import make_bandit
from offline_world.bandit_ensemble import LearnedBanditEnv
from experience.wrapper import ActionRewardWrapper
from experience.collector import BanditCollector
from experience.evaluator import BanditEvaluator
from experience.agent_buffer import TapeBuffer

from memory.lru import StackedLRU
from online_rl.q_modules import (
    RecurrentQNetwork,
    GreedyAgent,
    EpsilonGreedyAgent,
)
from online_rl.common import get_optimizer
from online_rl.q_losses import ddqn_update

eval_probs = [0.01, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.99]


def load_eval_env(config):

    def make_env(eval_prob):
        def _thunk():
            env = make_bandit(p_list=[0.5, eval_prob])
            return ActionRewardWrapper(env)

        return _thunk

    env_fns = [
        make_env(eval_prob)
        for eval_prob in eval_probs
        for _ in range(config["real_episodes"])
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)
    # |eval_probs| * config["real_episodes"] envs in total
    return envs


def make_buffer(buffer_size, obs_shape):
    return TapeBuffer(
        buffer_size,
        "start",
        {
            "observation": {
                "shape": obs_shape,
                "dtype": np.float32,
            },
            "action": {"shape": (), "dtype": np.int32},
            "next_reward": {"shape": (), "dtype": np.float32},
            "start": {"shape": (), "dtype": bool},
            "next_terminated": {"shape": (), "dtype": bool},
            "next_truncated": {"shape": (), "dtype": bool},
        },
    )


@hydra.main(version_base=None, config_path="configs", config_name="offline_bandit")
def main(cfg: DictConfig):
    # Convert to dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)

    if config.get("debug", False):
        # jax.config.update("jax_disable_jit", True)
        os.environ["WANDB_MODE"] = "disabled"

    name = datetime.now().strftime("%m-%d-%H-%M-%S")
    name += f'-{config["seed"]}'
    wandb.init(
        project="offline_bandit",
        name=name,
        config=config,
    )

    """
    Setup evaluation
    """
    key = random.PRNGKey(config["seed"])
    config["eval"]["seed"] = config["seed"] + 1000
    eval_key = random.PRNGKey(config["eval"]["seed"])

    eval_envs = load_eval_env(config["eval"])
    evaluator = BanditEvaluator(eval_envs)

    obs_shape = eval_envs.envs[0].observation_space.shape
    act_shape = eval_envs.envs[0].action_space.n
    max_horizon = eval_envs.envs[0].max_episode_steps
    print(
        "obs_shape",
        obs_shape,
        "act_shape",
        act_shape,
        "max_horizon",
        max_horizon,
        jax.default_backend(),
    )

    """
    Setup world model (training envs) and collector
    """
    train_envs = LearnedBanditEnv(original_env=make_bandit(p_list=[0.5, 0.5]))
    train_envs.load_model_and_data(model_seed=config["seed"], **config["collect"])
    collector = BanditCollector(train_envs)

    """
    Setup agent
    """
    key, model_key, memory_key = random.split(key, 3)
    memory_network = StackedLRU(**config["model"]["memory"], key=memory_key)
    memory_target = StackedLRU(**config["model"]["memory"], key=memory_key)
    q_network = RecurrentQNetwork(
        obs_shape, act_shape, memory_network, config["model"], model_key
    )
    q_target = eqx.nn.inference_mode(
        RecurrentQNetwork(
            obs_shape, act_shape, memory_target, config["model"], model_key
        )
    )

    """
    Setup optimizers
    """

    opt_enc = get_optimizer(
        lr=config["train"]["enc_lr"],
        lr_ratio=config["train"]["lr_ratio"],
        config=config["train"],
    )
    opt_head = get_optimizer(
        lr=config["train"]["head_lr"],
        lr_ratio=config["train"]["lr_ratio"],
        config=config["train"],
    )

    def label_fn(params):
        # NOTE: assume RecurrentQNetwork follows this pytree structure
        # inspired by https://arxiv.org/abs/2405.15384
        labels = eqx.tree_at(lambda m: m.pre, params, "encoder")
        labels = eqx.tree_at(lambda m: m.memory, labels, "encoder")
        labels = eqx.tree_at(lambda m: m.head, labels, "critic")
        return labels

    opt = optax.partition(
        transforms={
            "encoder": opt_enc,
            "critic": opt_head,
        },
        param_labels=label_fn,
    )
    opt_state = opt.init(eqx.filter(q_network, eqx.is_inexact_array))

    rb = make_buffer(config["train"]["buffer_size"], obs_shape)

    env_steps = 0
    grad_steps = 0
    eval_times = -1
    eval_every_grad_step = config["train"]["grad_steps"] // config["eval"]["times"]
    pbar = tqdm(total=config["train"]["grad_steps"], desc="grad_steps")

    gamma = jnp.float32(config["train"]["gamma"])
    tau = jnp.float32(config["train"]["target_tau"])
    train_start = time.time()

    while True:
        # Eval
        if grad_steps // eval_every_grad_step > eval_times:
            eval_times = grad_steps // eval_every_grad_step
            q_eval = eqx.filter_jit(eqx.nn.inference_mode)(q_network)
            eval_key, rollout_key, real_key = random.split(eval_key, 3)

            _, train_envs_metrics = collector(
                GreedyAgent(q_eval),
                parallel_size=config["eval"]["rollout_episodes"],
                report=True,
                key=rollout_key,
            )

            real_data, real_metrics = evaluator(
                GreedyAgent(q_eval),
                real_key,
            )
            to_log = {
                "count/eval_times": eval_times,
                "count/env_step": env_steps,
                "count/grad_step": grad_steps,
                "count/wall_min": (time.time() - train_start) / 60,
                "count/enc_lr": opt_state[0]["encoder"][0][-1].hyperparams[
                    "learning_rate"
                ],
                "count/head_lr": opt_state[0]["critic"][0][-1].hyperparams[
                    "learning_rate"
                ],
                **{"train_envs_eval/" + k: v for k, v in train_envs_metrics.items()},
                **{
                    "test_envs/" + k: v
                    for k, v in real_metrics.items()
                    if k != "reward"
                },
            }

            per_real_reward = (
                real_metrics["reward"].reshape(len(eval_probs), -1).mean(-1)
            )
            for i in range(len(eval_probs)):
                # arm identification for each possible bandit problem
                eval_actions = np.array(
                    [
                        traj["action"]
                        for traj in real_data[
                            i
                            * config["eval"]["real_episodes"] : (i + 1)
                            * config["eval"]["real_episodes"]
                        ]
                    ]
                )
                best_arm = eval_probs[i] > 0.5
                best_arm_rate = np.mean(eval_actions[:, :-1] == best_arm)
                # perf for each possible bandit problem
                to_log.update(
                    {
                        f"test_envs_reward/{eval_probs[i]}": per_real_reward[i],
                        f"best_arm_rate/{eval_probs[i]}": best_arm_rate,
                    }
                )

            if grad_steps > 0:
                to_log.update(
                    {
                        "count/epsilon": agent.epsilon,
                        "count/progress": progress,
                        **{
                            "train_envs_rollout/" + k: v
                            for k, v in rollout_metrics.items()
                        },
                        **{"agent/" + k: v for k, v in train_metrics.items()},
                    }
                )
            wandb.log(to_log)

        if grad_steps > config["train"]["grad_steps"]:
            break

        progress = jnp.float32(
            min(
                grad_steps
                / (config["train"]["grad_steps"] * config["collect"]["eps_end_ratio"]),
                1.0,
            )
        )
        agent = EpsilonGreedyAgent(
            q_network,
            eps_start=config["collect"]["eps_start"],
            eps_end=config["collect"]["eps_end"],
            progress=progress,
        )
        key, collect_key = random.split(key)
        rollout_data, rollout_metrics = collector(
            agent,
            parallel_size=config["collect"]["parallel_size"],
            report=False,
            key=collect_key,
        )
        for transitions in rollout_data:
            rb.add(**transitions)

        env_steps += rollout_metrics["data_size"]
        delta_grad_step = math.ceil(
            rollout_metrics["data_size"] * config["train"]["utd"]
        )
        grad_steps += delta_grad_step
        pbar.update(delta_grad_step)

        for _ in tqdm(range(delta_grad_step)):
            key, sample_key, loss_key = random.split(key, 3)
            data = rb.sample(config["train"]["batch_size"], sample_key)
            data = jax.tree.map(lambda x: jnp.asarray(x), data)
            (
                q_network,
                q_target,
                opt_state,
                train_metrics,
            ) = ddqn_update(
                q_network,
                q_target,
                data,
                opt,
                opt_state,
                gamma,
                tau,
                loss_key,
            )


if __name__ == "__main__":
    main()
