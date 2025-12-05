"""
Ensemble world model trained on the bandit data.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import copy
import os, json, glob

from offline_world.bandit_data import make_bandit
from offline_world.modules import (
    EnsembleBanditModel,
    update_selected_members,
    subselect_members,
)
from offline_world.losses import get_nll_and_unc, get_train_loss


@eqx.filter_jit
def _reset(parallel_size: int, num_actions: int):
    obs = jnp.zeros((parallel_size, 1))  # (N*B, 1)
    null_action = jax.nn.one_hot(0, num_actions, dtype=jnp.float32)
    null_action = jnp.tile(null_action, (parallel_size, 1))  # (N*B, A)
    null_reward = jnp.zeros((parallel_size, 1), dtype=jnp.float32)  # (N*B, 1)
    return jnp.concatenate([obs, null_action, null_reward], axis=1)  # (N*B, d)


@eqx.filter_jit
def _step(
    ensemble: EnsembleBanditModel,
    num_actions: int,
    action: jax.Array,
    unc_type: str,
    penalty_coef: float,
    key: jax.random.PRNGKey,
):
    action_onehot = jax.nn.one_hot(action, num_actions, dtype=jnp.float32)  # (N*B, A)
    _, unc_dict = ensemble.forward_same_data(action_onehot)
    unc = unc_dict[unc_type]  # (N*B)

    diff_action = action_onehot.reshape(
        ensemble.ensemble_size, -1, num_actions
    )  # (N, B, A)
    mu, sigma = ensemble.forward_diff_data(diff_action)  # (N, B, 1)
    mu, sigma = mu.flatten(), sigma.flatten()  # (N*B), (N*B)

    eps = jax.random.normal(key, mu.shape)
    raw_reward = mu + eps * sigma
    reward = raw_reward - penalty_coef * unc

    obs = jnp.zeros((action.shape[0], 1))
    # NOTE: to align with inference-time reward, no need for using world model
    return (
        jnp.concatenate([obs, action_onehot, jnp.expand_dims(raw_reward, 1)], axis=1),
        reward,
    )


class LearnedBanditEnv:
    """
    This is the class for the learned bandit environment, with two main functions:
    - learning the reward models from a dataset before RL training
    - simulating like a vectorized gym environment for the RL agent to interact with
    """

    def __init__(
        self,
        original_env,
    ):
        # these quantities are known to the agent
        self.horizon = original_env._max_episode_steps
        self.num_actions = int(original_env.action_space.n)
        assert original_env.observation_space.shape == (1,)

    def load_data(self, data_path: str):
        import pickle
        from sklearn.model_selection import train_test_split

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            # print("data", data)

        ## extract the (actions, rewards) from the dataset
        # omit the dummy final timestep
        actions = np.concatenate([d["action"][:-1] for d in data], axis=0)
        rewards = np.concatenate([d["next_reward"][:-1] for d in data], axis=0)
        actions = jax.nn.one_hot(actions, self.num_actions)  # (B, A)
        rewards = jnp.expand_dims(rewards, 1)  # (B, 1)

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            actions, rewards, test_size=0.2, random_state=42
        )  # here we make sure the split is deterministic
        print(actions.shape, rewards.shape, "reward mean", rewards.mean())
        print("valid reward mean/std", Y_valid.mean(), Y_valid.std())

        return X_train, X_valid, Y_train, Y_valid

    def train_model(
        self,
        data_path: str,
        save_dir: str,
        seed: int,
        ensemble_size: int = 128,
        hidden_size: int = 16,
        total_epochs: int = 500,
        max_epochs_since_update: int = 5,
        improve_thres: float = 0.001,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
    ):
        """
        This function is only called by running this file before RL training.
        Should be very fast to run (~1 min)
        """
        cfg = {k: v for k, v in locals().items() if k != "self"}  # current arguments

        import optax
        from datetime import datetime
        import wandb

        X_train, X_valid, Y_train, Y_valid = self.load_data(data_path)

        name = datetime.now().strftime("%m-%d-%H-%M-%S")
        name += f"-{seed}"
        wandb.init(
            project="ensemble_bandit",
            name=name,
            config=cfg,
        )

        key = jax.random.PRNGKey(seed)
        key, model_key = jax.random.split(key)

        self.ensemble = EnsembleBanditModel(
            ensemble_size=ensemble_size,
            input_size=self.num_actions,
            hidden_size=hidden_size,
            has_ln=True,
            key=model_key,
        )
        self.visualize(save_dir, seed, "init")

        opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        opt_state = opt.init(eqx.filter(self.ensemble, eqx.is_inexact_array))

        @eqx.filter_jit
        def train_step(model, opt_state, x, y):
            loss, grads = get_train_loss(model, x, y)
            updates, opt_state = opt.update(
                grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        best_loss, unc, _ = get_nll_and_unc(
            eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
        )
        wandb.log({"epoch": 0, "loss_best": best_loss.mean(), **unc})

        best_ensemble = copy.deepcopy(self.ensemble)
        epochs_wait = 0
        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(total_epochs):
            # shuffle the data for each model independently
            key, perm_key = jax.random.split(key)
            head_keys = jax.random.split(perm_key, ensemble_size)
            perm_heads = jax.vmap(lambda k: jax.random.permutation(k, num_samples))(
                head_keys
            )

            for i in range(num_batches):
                batch_indices = perm_heads[:, i * batch_size : (i + 1) * batch_size]
                self.ensemble, opt_state, loss = train_step(
                    self.ensemble,
                    opt_state,
                    X_train[batch_indices],
                    Y_train[batch_indices],
                )

            # eval the ensemble
            loss_valid, unc, _ = get_nll_and_unc(
                eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
            )

            # update the best ensemble components
            improved = (best_loss - loss_valid) > improve_thres  # bool mask (N,)

            if jnp.any(improved):
                best_ensemble = update_selected_members(
                    best_ensemble, self.ensemble, improved
                )
                best_loss = best_loss.at[improved].set(loss_valid[improved])
                epochs_wait = 0
            else:
                epochs_wait += 1

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "loss_train": loss,
                    "loss_valid": loss_valid.mean(),
                    "loss_best": best_loss.mean(),
                    **unc,
                }
            )

            if epochs_wait >= max_epochs_since_update:
                print("Early stopping at epoch", epoch)
                break

        ## save the best model
        self.ensemble = copy.deepcopy(best_ensemble)
        self.visualize(save_dir, seed, "final")

        loss_valid, unc, _ = get_nll_and_unc(
            eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
        )
        print(f"final loss_valid = {loss_valid.mean()}, {unc = }")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ensemble_seed{seed}.eqx")
        with open(save_path, "wb") as f:
            hparam_str = json.dumps(
                {
                    "ensemble_size": ensemble_size,
                    "hidden_size": hidden_size,
                }
            )
            f.write((hparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.ensemble)
            print(f"saved model to {save_path}")

    def visualize(self, save_dir: str, seed: int, desc: str):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        sns.set_theme(context="talk", style="whitegrid")

        ## sanity check the one-hot inputs
        x = jnp.eye(self.num_actions)
        (mu, sigma), unc = self.ensemble.forward_same_data(x)
        mu, sigma = mu.squeeze(), sigma.squeeze()
        print(f"{mu.mean(0) = }, {mu.std(0) = }")
        print(f"{sigma.mean(0) = }, {sigma.std(0) = }")
        print(unc)

        arms = jnp.tile(jnp.arange(self.num_actions), self.ensemble.ensemble_size)
        data_mean = pd.DataFrame({"r_mean": jnp.clip(mu, -1, 1).flatten(), "arm": arms})

        plt.figure(figsize=(4, 4))
        sns.histplot(data=data_mean, x="r_mean", hue="arm", binwidth=0.03)
        plt.xlim(0, 1)  # focus on bernoulli assumption
        plt.title("Histogram of Reward Mean")
        plt.xlabel("Reward Mean")

        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"seed{seed}_{desc}.pdf")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"saved figure to {fig_path}")
        plt.close()

    def load_model_and_data(
        self,
        data_path: str,
        save_dir: str,
        model_seed: int,
        ensemble_size: int,  # the actual ensemble size used in rollouts, i.e., N
        penalty_coef: float,  # the penalty coefficient for the rollout
        unc_type: str = "epi_mean",
        **kwargs,
    ):
        """
        Called by the main script as initialization.
        """
        ## load the model
        all_paths = sorted(glob.glob(os.path.join(save_dir, f"ensemble_seed*.eqx")))
        seed = model_seed % len(all_paths)
        model_path = all_paths[seed]

        key = jax.random.PRNGKey(seed + 128)
        key, model_key = jax.random.split(key)

        with open(model_path, "rb") as f:
            hparams = json.loads(f.readline().decode())
            random_ensemble = EnsembleBanditModel(
                ensemble_size=hparams["ensemble_size"],
                input_size=self.num_actions,
                hidden_size=hparams["hidden_size"],
                has_ln=True,
                key=model_key,
            )
            ensemble = eqx.tree_deserialise_leaves(f, random_ensemble)
            print(f"loaded model from {model_path}")

        ### select the subset of ensemble members
        _, X_valid, _, Y_valid = self.load_data(data_path)

        loss_valid, _, _ = get_nll_and_unc(
            eqx.nn.inference_mode(ensemble), X_valid, Y_valid
        )
        best_idx = jnp.argsort(loss_valid)[:ensemble_size]
        print(f"{best_idx = }, {best_idx.shape = }")

        self.ensemble = subselect_members(ensemble, best_idx)
        self.visualize(save_dir, seed, f"final{ensemble_size}")

        assert unc_type in ["epi_mean", "ale_max", "total_var"]
        self.unc_type = unc_type
        self.penalty_coef = penalty_coef

    def reset(self, parallel_size: int, key: jax.random.PRNGKey):
        """
        Interface as a Jax-based vectorized environment:
            rollout parallel_size (i.e., N*B) envs in parallel,
            interally it assigns B envs to each ensemble member.
        Return: observation (N*B, 1+A+1)
        """
        assert parallel_size % self.ensemble.ensemble_size == 0
        self.parallel_size = parallel_size
        self.t = jnp.zeros((self.parallel_size,), dtype=jnp.int32)

        return _reset(self.parallel_size, self.num_actions)

    def step(self, action: jax.Array, key: jax.random.PRNGKey):
        """
        Interface as a Jax-based vectorized environment:
        action: (N*B,)
        Return: observation (N*B, 1+A+1), reward (N*B,)
        """
        observation, reward = _step(
            ensemble=eqx.nn.inference_mode(self.ensemble),
            num_actions=self.num_actions,
            action=action,
            unc_type=self.unc_type,
            penalty_coef=self.penalty_coef,
            key=key,
        )

        self.t += 1
        truncated = self.t >= self.horizon
        terminated = jnp.zeros((self.parallel_size,), dtype=bool)

        return observation, reward, terminated, truncated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="seed for the model")
    args = parser.parse_args()

    ## train the model
    world = LearnedBanditEnv(
        original_env=make_bandit([0.5, 0.5]),
    )
    world.train_model(
        data_path="offline_world/data/bandit.pkl",
        save_dir="offline_world/ckpt/bandit/",
        seed=args.seed,
    )

    ## load the model
    # world = LearnedBanditEnv(
    #     original_env=make_bandit([0.5, 0.5]),
    # )
    # world.load_model_and_data(
    #     data_path="offline_world/data/bandit.pkl",
    #     save_dir="offline_world/ckpt/bandit/",
    #     model_seed=args.seed,
    #     ensemble_size=100,
    #     parallel_size=100,
    #     penalty_coef=0.0,
    # )
    # world.reset()
