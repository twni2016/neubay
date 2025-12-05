from jax import numpy as jnp
from jax import random
import jax
import numpy as np
import math
import time, os, json
from datetime import datetime
import equinox as eqx
import optax
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import gym
from offline_world.cont_ensemble import LearnedContEnv
from experience.wrapper import make_env, ActionRewardWrapper
from experience.collector import ContCollector
from experience.evaluator import ContEvaluator
from experience.agent_buffer import TapeBuffer

from memory.lru import StackedLRU
from online_rl.ac_modules import (
    RecurrentCritic,
    RecurrentActor,
    Alpha,
)
from online_rl.common import get_optimizer
from online_rl.ac_losses import sac_update


def load_env(domain: str, dataset_name: str, parallel_size: int):
    def make():
        def _thunk():
            env = make_env(domain, dataset_name)
            env = ActionRewardWrapper(env)
            return env

        return _thunk

    env_fns = [make() for _ in range(parallel_size)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    return envs


def make_buffer(buffer_size, obs_shape, act_shape):
    return TapeBuffer(
        buffer_size,
        "start",
        {
            "observation": {
                "shape": obs_shape,
                "dtype": np.float32,
            },
            "action": {"shape": act_shape, "dtype": np.float32},
            "next_reward": {"shape": (), "dtype": np.float32},
            "start": {"shape": (), "dtype": bool},
            "next_terminated": {"shape": (), "dtype": bool},
            "next_truncated": {"shape": (), "dtype": bool},
            # used in MBRL, whether transition is from real env or world model
            "real": {"shape": (), "dtype": bool},
        },
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Convert to dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)

    if config.get("debug", False):
        # jax.config.update("jax_disable_jit", True)
        os.environ["WANDB_MODE"] = "disabled"

    name = datetime.now().strftime("%m-%d-%H-%M-%S")
    name += f'-{config["seed"]}'
    wandb.init(
        project=config["domain"],
        name=name,
        config=config,
    )

    """
    Setup evaluation
    """
    key = random.PRNGKey(config["seed"])
    eval_key = random.PRNGKey(config["seed"] + 1000)

    eval_envs = load_env(
        config["domain"], config["dataset_name"], config["eval"]["real_episodes"]
    )
    evaluator = ContEvaluator(eval_envs)

    obs_shape = eval_envs.envs[0].observation_space.shape
    act_shape = eval_envs.envs[0].action_space.shape
    max_horizon = eval_envs.envs[0].max_episode_steps
    assert np.all(eval_envs.envs[0].action_space.high == 1.0)
    assert np.all(eval_envs.envs[0].action_space.low == -1.0)
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
    train_envs = LearnedContEnv()
    train_envs.load_model_and_data(
        domain=config["domain"],
        dataset_name=config["dataset_name"],
        save_dir=config["ensemble"]["save_dir"],
        plot_dir=config["ensemble"]["plot_dir"],
        model_seed=config["seed"],
        **config["collect"],
    )
    collector = ContCollector(train_envs)

    """
    Setup agent
    """
    key, m1_key, m2_key, q_key, pi_key = random.split(key, 5)
    q_memory_network = StackedLRU(**config["model"]["memory"], key=m1_key)
    q_memory_target = StackedLRU(**config["model"]["memory"], key=m1_key)
    pi_memory_network = StackedLRU(**config["model"]["memory"], key=m2_key)
    q_network = RecurrentCritic(
        obs_shape, act_shape, q_memory_network, config["model"], q_key
    )
    q_target = eqx.nn.inference_mode(
        RecurrentCritic(obs_shape, act_shape, q_memory_target, config["model"], q_key)
    )
    pi_network = RecurrentActor(
        obs_shape, act_shape, pi_memory_network, config["model"], pi_key
    )
    if config["model"]["target_alpha_ratio"] is None:
        target_ent = None
    else:
        target_ent = -float(np.prod(act_shape)) * config["model"]["target_alpha_ratio"]
    alpha = Alpha(init_value=config["model"]["init_alpha"], target_entropy=target_ent)

    """
    Setup optimizers
    """

    opt_critic_enc = get_optimizer(
        lr=config["train"]["critic_enc_lr"],
        lr_ratio=config["train"]["critic_lr_ratio"],
        config=config["train"],
    )
    opt_critic_head = get_optimizer(
        lr=config["train"]["critic_head_lr"],
        lr_ratio=config["train"]["critic_lr_ratio"],
        config=config["train"],
    )

    def critic_label_fn(params):
        # NOTE: assume RecurrentCritic follows this pytree structure
        # inspired by https://arxiv.org/abs/2405.15384
        labels = eqx.tree_at(lambda m: m.pre, params, "encoder")
        labels = eqx.tree_at(lambda m: m.memory, labels, "encoder")
        labels = eqx.tree_at(lambda m: m.critic_heads, labels, "critic")
        return labels

    opt_q = optax.partition(
        transforms={
            "encoder": opt_critic_enc,
            "critic": opt_critic_head,
        },
        param_labels=critic_label_fn,
    )
    opt_state_q = opt_q.init(eqx.filter(q_network, eqx.is_inexact_array))

    opt_actor_enc = get_optimizer(
        lr=config["train"]["actor_enc_lr"],
        lr_ratio=config["train"]["actor_lr_ratio"],
        config=config["train"],
    )
    opt_actor_head = get_optimizer(
        lr=config["train"]["actor_head_lr"],
        lr_ratio=config["train"]["actor_lr_ratio"],
        config=config["train"],
    )

    def actor_label_fn(params):
        # NOTE: assume RecurrentActor follows this pytree structure
        # inspired by https://arxiv.org/abs/2405.15384
        labels = eqx.tree_at(lambda m: m.pre, params, "encoder")
        labels = eqx.tree_at(lambda m: m.memory, labels, "encoder")
        labels = eqx.tree_at(lambda m: m.actor_head, labels, "actor")
        return labels

    opt_pi = optax.partition(
        transforms={
            "encoder": opt_actor_enc,
            "actor": opt_actor_head,
        },
        param_labels=actor_label_fn,
    )
    opt_state_pi = opt_pi.init(eqx.filter(pi_network, eqx.is_inexact_array))

    if config["model"]["target_alpha_ratio"] is None:
        opt_alpha = opt_state_alpha = None
    else:
        opt_alpha = optax.adam(learning_rate=config["train"]["alpha_lr"])
        opt_state_alpha = opt_alpha.init(eqx.filter(alpha, eqx.is_inexact_array))

    """
    Start training
    """
    rb = make_buffer(config["train"]["buffer_size"], obs_shape, act_shape)

    env_steps = 0
    grad_steps = 0
    eval_times = -1
    eval_every_grad_step = config["train"]["grad_steps"] // config["eval"]["times"]
    pbar = tqdm(total=config["train"]["grad_steps"], desc="grad_steps")

    gamma = jnp.float32(config["train"]["gamma"])
    tau = jnp.float32(config["train"]["target_tau"])
    real_weight = jnp.float32(config["train"]["real_weight"])
    backup_entropy = config["train"].get("backup_entropy", True)
    train_start = time.time()

    ## FOR DEBUGGING ANTMAZE
    if "antmaze" in config["domain"]:
        from experience.visualize_antmaze import setup_maze, plot_trajs

        setup_maze(config["dataset_name"])

    while True:
        # Eval
        if grad_steps // eval_every_grad_step > eval_times:
            eval_times = grad_steps // eval_every_grad_step
            eval_key, real_key = random.split(eval_key, 2)
            real_rollouts, real_metrics = evaluator(
                eqx.nn.inference_mode(pi_network), deterministic=True, key=real_key
            )
            to_log = {
                "count/eval_times": eval_times,
                "count/env_step": env_steps,
                "count/grad_step": grad_steps,
                "count/wall_min": (time.time() - train_start) / 60,
                "count/critic_enc_lr": opt_state_q[0]["encoder"][0][-1].hyperparams[
                    "learning_rate"
                ],
                "count/critic_head_lr": opt_state_q[0]["critic"][0][-1].hyperparams[
                    "learning_rate"
                ],
                "count/actor_enc_lr": opt_state_pi[0]["encoder"][0][-1].hyperparams[
                    "learning_rate"
                ],
                "count/actor_head_lr": opt_state_pi[0]["actor"][0][-1].hyperparams[
                    "learning_rate"
                ],
                **{"test_envs/" + k: v for k, v in real_metrics.items()},
            }

            if "antmaze" in config["domain"]:
                to_log["eval/trajectories"] = plot_trajs(real_rollouts, eval_times)

            if grad_steps > 0:
                to_log.update(
                    {
                        **{"train_envs/" + k: v for k, v in rollout_metrics.items()},
                        **{"agent/" + k: v for k, v in train_metrics.items()},
                    }
                )
            wandb.log(to_log)

        if grad_steps > config["train"]["grad_steps"]:
            break

        # Collect data
        key, collect_key = random.split(key, 2)
        rollout_data, rollout_metrics = collector(
            eqx.nn.inference_mode(pi_network),
            deterministic=False,
            start_from_s0=False,
            parallel_size=config["collect"]["parallel_size"],
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
                pi_network,
                alpha,
                q_target,
                opt_state_q,
                opt_state_pi,
                opt_state_alpha,
                train_metrics,
            ) = sac_update(
                q_network=q_network,
                pi_network=pi_network,
                alpha=alpha,
                q_target=q_target,
                data=data,
                opt_q=opt_q,
                opt_pi=opt_pi,
                opt_alpha=opt_alpha,
                opt_state_q=opt_state_q,
                opt_state_pi=opt_state_pi,
                opt_state_alpha=opt_state_alpha,
                gamma=gamma,
                tau=tau,
                real_weight=real_weight,
                backup_entropy=backup_entropy,
                loss_key=loss_key,
            )

    if config["save_agent"]:
        # save the last agent checkpoint
        save_dir = os.path.join(
            os.path.expanduser(config["save_agent_root"]),
            config["domain"],
            config["dataset_name"],
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"agent_seed{config['seed']}.eqx")

        with open(save_path, "wb") as f:
            f.write((json.dumps(config) + "\n").encode())
            eqx.tree_serialise_leaves(
                f,
                {
                    "q_network": q_network,
                    "q_target": q_target,
                    "pi_network": pi_network,
                    "alpha": alpha,
                },
            )
            print(f"saved agent to {save_path}")


if __name__ == "__main__":
    main()
