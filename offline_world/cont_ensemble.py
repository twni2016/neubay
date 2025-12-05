"""
Ensemble world model trained on the continuous control data.
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import copy
import os, json, glob
from datetime import datetime
import time
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import gym, d4rl
from experience.wrapper import make_env
from experience.world_buffer import world_learning_dataset, HistorySampler
from offline_world.modules import (
    EnsembleContModel,
    update_selected_members,
    subselect_members,
    Scaler,
)
from offline_world.losses import get_mse_and_unc, get_train_loss
from sklearn.model_selection import train_test_split
from offline_world.static_fns import get_termination_fn, OracleEnv

np.set_printoptions(precision=4, suppress=True)


@eqx.filter_jit
def _step(
    ensemble: EnsembleContModel,
    state_action: jax.Array,  # (N*B, S+A)
    unc_type: str,
    key: jax.random.PRNGKey,
):
    _, unc_dict = ensemble.forward_same_data(state_action)
    unc = unc_dict[unc_type]  # (N*B)

    diff_state_action = state_action.reshape(
        ensemble.ensemble_size, -1, state_action.shape[-1]
    )  # (N, B, S+A)
    mu, sigma = ensemble.forward_diff_data(diff_state_action)  # (N, B, S+1)
    eps = jax.random.normal(key, mu.shape)
    next_state_reward = (mu + eps * sigma).reshape(-1, mu.shape[-1])  # (N*B, S+1)

    return next_state_reward, unc


class LearnedContEnv:
    """
    This is the class for the learned continuous environment, with two main functions:
    - learning the reward models from a dataset before RL training
    - simulating like a vectorized gym environment for the RL agent to interact with
    """

    def load_world_learning_data(self, domain: str, env):
        """
        For ensemble training, we only need Markovian transition tuples
            (obs, act, next_obs, reward)
        """
        data = world_learning_dataset(domain, env)

        self.scaler = Scaler(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
        )
        raw_inputs = np.concatenate([data["observations"], data["actions"]], axis=-1)
        raw_targets = np.concatenate(
            [data["next_observations"], data["rewards"][:, None]], axis=-1
        )
        inputs = self.scaler.transform_inputs(raw_inputs)
        targets = self.scaler.transform_outputs(raw_targets)
        print(
            f"{inputs.shape=}, {targets.shape=}, {inputs.mean(0)=}, {targets.mean(0)=}"
        )

        holdout_size = min(int(inputs.shape[0] * 0.2), 1000)
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            inputs, targets, test_size=holdout_size, random_state=42
        )  # here we make sure the split is deterministic to compare model performance
        print(f"{X_train.shape=}, {Y_train.shape=}, {X_valid.shape=}, {Y_valid.shape=}")

        return X_train, X_valid, Y_train, Y_valid

    def train_model(
        self,
        domain: str,
        dataset_name: str,
        seed: int,
        save_dir: str,
        plot_dir: str,
        total_size: int = 128,
        hidden_size: int = 200,  # 4 hidden layers of this size
        has_ln: bool = True,  # layernorm in model
        total_epochs: int = None,
        max_epochs_since_update: int = 5,
        improve_thres: float = 0.01,
        batch_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.00001,
    ):
        """
        This function is only called by running this file before RL training.
        """
        cfg = {k: v for k, v in locals().items() if k != "self"}  # current arguments

        original_env = make_env(domain, dataset_name)
        # these quantities are known to the agent
        self.state_dim = original_env.observation_space.shape[0]
        self.act_dim = original_env.action_space.shape[0]
        assert np.all(original_env.action_space.low == -1.0)
        assert np.all(original_env.action_space.high == 1.0)

        name = datetime.now().strftime("%m-%d-%H-%M-%S")
        name += f"-{seed}"
        wandb.init(
            project=domain,
            name=name,
            config=cfg,
        )

        X_train, X_valid, Y_train, Y_valid = self.load_world_learning_data(
            domain, original_env
        )

        key = jax.random.PRNGKey(seed)
        key, model_key = jax.random.split(key)

        self.ensemble = EnsembleContModel(
            ensemble_size=total_size,
            obs_dim=self.state_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
            has_ln=has_ln,
            key=model_key,
        )

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

        num_samples = X_train.shape[0]
        mse_best, unc_valid, _ = get_mse_and_unc(
            eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
        )
        key, eval_key = jax.random.split(key)
        train_indices = jax.random.randint(
            eval_key, len(X_valid), minval=0, maxval=num_samples
        )
        unc_train = get_mse_and_unc(
            eqx.nn.inference_mode(self.ensemble),
            X_train[train_indices],
            Y_train[train_indices],
        )[1]
        wandb.log(
            {
                "epoch": 0,
                "mse_best": mse_best.mean(),
                **{"unc/" + k: v for k, v in unc_valid.items()},
                **{
                    "unc_diff/" + k: unc_train[k] - unc_valid[k]
                    for k in unc_valid.keys()
                },
            }
        )

        best_ensemble = copy.deepcopy(self.ensemble)
        epoch = 0
        epochs_wait = 0
        num_batches = (num_samples + batch_size - 1) // batch_size
        train_start = time.time()

        while True:
            # shuffle the data for each model independently
            key, perm_key = jax.random.split(key)
            head_keys = jax.random.split(perm_key, total_size)
            perm_heads = jax.vmap(lambda k: jax.random.permutation(k, num_samples))(
                head_keys
            )  # this line takes a lot GPU memory, but worth it for speed-up
            for i in tqdm(range(num_batches), desc=f"epoch {epoch}"):
                batch_indices = perm_heads[:, i * batch_size : (i + 1) * batch_size]
                self.ensemble, opt_state, loss = train_step(
                    self.ensemble,
                    opt_state,
                    X_train[batch_indices],
                    Y_train[batch_indices],
                )

            # eval the ensemble
            mse_valid, unc_valid, _ = get_mse_and_unc(
                eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
            )
            unc_train = get_mse_and_unc(
                eqx.nn.inference_mode(self.ensemble),
                X_train[train_indices],
                Y_train[train_indices],
            )[1]

            # update the best ensemble components, bool mask (N,)
            improved = ((mse_best - mse_valid) / mse_best) > improve_thres

            if jnp.any(improved):
                best_ensemble = update_selected_members(
                    best_ensemble, self.ensemble, improved
                )
                mse_best = mse_best.at[improved].set(mse_valid[improved])
                epochs_wait = 0
            else:
                epochs_wait += 1

            epoch += 1
            wandb.log(
                {
                    "epoch": epoch,
                    "wall_min": (time.time() - train_start) / 60,
                    "loss_train": loss,
                    "mse_valid": mse_valid.mean(),
                    "mse_best": mse_best.mean(),
                    "num_improved": jnp.sum(improved),
                    **{"unc/" + k: v for k, v in unc_valid.items()},
                    **{
                        "unc_diff/" + k: unc_train[k] - unc_valid[k]
                        for k in unc_valid.keys()
                    },
                }
            )

            if epochs_wait >= max_epochs_since_update or (
                total_epochs is not None and epoch >= total_epochs
            ):
                print("Early stopping at epoch", epoch)
                break

        ## save the best model
        self.ensemble = copy.deepcopy(best_ensemble)

        mse_valid, unc_valid, _ = get_mse_and_unc(
            eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
        )
        print(f"final {mse_valid.mean() = }, {unc_valid = }")

        save_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ensemble_seed{seed}.eqx")
        with open(save_path, "wb") as f:
            hparam_str = json.dumps(
                {
                    "ensemble_size": total_size,
                    "hidden_size": hidden_size,
                    "has_ln": has_ln,
                }
            )
            f.write((hparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.ensemble)
            print(f"saved model to {save_path}")

    def load_model_and_data(
        self,
        domain: str,
        dataset_name: str,
        save_dir: str,
        plot_dir: str,
        model_seed: int,
        ensemble_size: int,  # the actual ensemble size used in rollouts, i.e., N
        penalty_coef: float,  # the normalized penalty coefficient for the rollout
        unc_quantile: float,  # in [0,1], while -1 means not using unc truncation
        unc_type: str = "epi_mean",
        max_rollout_len: int = -1,  # -1 means not using max rollout length
        **kwargs,
    ):
        """
        Called by the main script as initialization.
        """
        original_env = make_env(domain, dataset_name)
        # these quantities are known to the agent
        self.max_horizon = original_env._max_episode_steps
        self.state_dim = original_env.observation_space.shape[0]
        self.act_dim = original_env.action_space.shape[0]

        ## 1. load the model
        model_dir = os.path.join(save_dir, dataset_name)
        all_paths = sorted(glob.glob(os.path.join(model_dir, f"ensemble_seed*.eqx")))
        seed = model_seed % len(all_paths)
        model_path = all_paths[seed]
        self.plot_dir = os.path.join(
            plot_dir,
            dataset_name,
            model_path[model_path.find("ensemble_seed") :].replace(".eqx", ""),
        )

        key = jax.random.PRNGKey(seed + 128)
        key, model_key = jax.random.split(key)

        with open(model_path, "rb") as f:
            hparams = json.loads(f.readline().decode())
            random_ensemble = EnsembleContModel(
                ensemble_size=hparams["ensemble_size"],
                obs_dim=self.state_dim,
                act_dim=self.act_dim,
                hidden_size=hparams["hidden_size"],
                has_ln=hparams["has_ln"],
                key=model_key,
            )
            ensemble = eqx.tree_deserialise_leaves(f, random_ensemble)
            print(f"\nloaded model from {model_path}")

        ## 2. select the subset of ensemble members
        X_train, X_valid, Y_train, Y_valid = self.load_world_learning_data(
            domain, original_env
        )

        mse_valid, _, _ = get_mse_and_unc(
            eqx.nn.inference_mode(ensemble), X_valid, Y_valid
        )
        best_idx = jnp.argsort(mse_valid)[:ensemble_size]
        print(f"{mse_valid.mean()=}, {mse_valid.max()=}, {mse_valid[best_idx].max()=}")
        self.ensemble = subselect_members(ensemble, best_idx)

        ## 3. get the uncertainty quantiles on offline dataset
        ### inspired by https://arxiv.org/abs/2304.04660 https://arxiv.org/abs/2405.19014
        assert unc_type in ["epi_mean", "ale_max", "total_var"]
        valid_unc = get_mse_and_unc(
            eqx.nn.inference_mode(self.ensemble), X_valid, Y_valid
        )[-1][unc_type]
        train_unc = []
        for i in tqdm(range(0, X_train.shape[0], len(X_valid)), desc="get train unc"):
            batch_unc = get_mse_and_unc(
                eqx.nn.inference_mode(self.ensemble),
                X_train[i : i + len(X_valid)],
                Y_train[i : i + len(X_valid)],
            )[-1][unc_type]
            train_unc.append(batch_unc)
        all_unc = jnp.concatenate(train_unc + [valid_unc], axis=0)
        self.avg_train_unc = all_unc.mean()

        if unc_quantile < 0:
            self.unc_thres = jnp.inf
            print(
                f"\nWarning: not using unc truncation, set unc_thres to {self.unc_thres}"
            )
        else:
            assert 0 <= unc_quantile <= 1
            self.unc_thres = jnp.quantile(all_unc, unc_quantile)
            print(f"\nusing {unc_quantile = }, set {self.unc_thres = }")

        ## 4. get training tools and configs
        self.history_sampler = HistorySampler(domain, original_env)
        self.termination_fn = get_termination_fn(dataset_name)
        self.unc_type = unc_type
        # the coef is calibrated by training unc
        self.penalty_coef = penalty_coef / self.avg_train_unc
        print(f"{self.avg_train_unc = :.3f}, {self.penalty_coef = :.5f}")

        self.max_rollout_len = (
            max_rollout_len if max_rollout_len > 0 else self.max_horizon
        )
        if max_rollout_len > 0:
            print(f"\nWarning: max_rollout_len is specified as {max_rollout_len}")

        ## below are only used for debugging
        self.oracle_env = OracleEnv(dataset_name)
        self.dataset_name = dataset_name
        self.unc_quantile = unc_quantile
        qs = jnp.array([0.9, 0.99, 0.999, 1.0])
        self.common_quantiles = {str(q): jnp.quantile(all_unc, q).item() for q in qs}
        print(
            f"\n{dataset_name} normalized quantiles:",
            ", ".join(
                f"{k}: {v / self.avg_train_unc:.3f}"
                for k, v in self.common_quantiles.items()
            ),
        )
        #### for the CDF plot
        # qs = jnp.linspace(0.0, 1.0, 10000)
        # uncs = jnp.quantile(all_unc, qs) / self.avg_train_unc
        # uncs = np.asarray(uncs)
        # np.save("offline_world/data/quantiles_" + dataset_name + ".npy", uncs)

    def reset(self, start_from_s0: bool, parallel_size: int, key: jax.random.PRNGKey):
        """
        Interface as a Jax-based vectorized environment:
            states: (N*B, S) where we will assign B states to each of the N ensemble members
            timesteps: (N*B,) the initial timesteps
        """
        histories, infos = self.history_sampler.sample(
            start_from_s0=start_from_s0, size=parallel_size, key=key
        )

        # extract the last padded observations as the initial states
        self.states = infos["padded_obs"][:, -1, : self.state_dim].copy()
        self.timesteps = infos["timesteps"].copy()
        self.truncateds = infos["truncateds"].copy()
        self.terminateds = infos["terminateds"].copy()

        return histories, infos

    def step(self, action: jax.Array, key: jax.random.PRNGKey):
        """
        Interface as a Jax-based vectorized environment:
        action: (N*B, A)
        Return: augmented observation (N*B, S+A+1), reward (N*B,),
                terminated (N*B,), truncated (N*B,), relative uncertainty (N*B,)
        """
        state_action = jnp.concatenate([self.states, action], axis=-1)  # (N*B, S+A)
        next_state_reward, unc = _step(
            ensemble=eqx.nn.inference_mode(self.ensemble),
            state_action=self.scaler.transform_inputs(state_action),  # normalize
            unc_type=self.unc_type,
            key=key,
        )
        next_state_reward = self.scaler.inverse_transform_outputs(next_state_reward)

        next_state, raw_reward = next_state_reward[:, :-1], next_state_reward[:, -1]
        next_obs = jnp.concatenate(
            [next_state, action, jnp.expand_dims(raw_reward, 1)], axis=-1
        )  # NOTE: to align with inference-time reward, no need for using world model
        reward = raw_reward - self.penalty_coef * unc

        ## get ground-truth terminals, once terminated, always terminated
        terminated = self.termination_fn(
            np.array(self.states), np.array(action), np.array(next_state)
        )
        self.terminateds = np.logical_or(self.terminateds, terminated)
        self.states = next_state.copy()

        ## get truncated flags, once truncated, always truncated
        self.timesteps += 1
        truncateds = np.logical_or(
            self.timesteps >= self.max_horizon,  # horizon extrapolation
            unc > self.unc_thres,  # uncertainty truncation
        )
        self.truncateds = np.logical_or(self.truncateds, truncateds)

        return (
            next_obs,
            reward,
            self.terminateds.copy(),
            self.truncateds.copy(),
            unc,
        )


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Convert to dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)

    if config.get("debug", False):
        # jax.config.update("jax_disable_jit", True)
        os.environ["WANDB_MODE"] = "disabled"

    world = LearnedContEnv()

    # ----------- train the model
    world.train_model(
        dataset_name=config["dataset_name"],
        seed=config["seed"],
        **config["ensemble"],
    )
    exit()

    # ----------- load the model
    world.load_model_and_data(
        domain=config["domain"],
        dataset_name=config["dataset_name"],
        save_dir=config["ensemble"]["save_dir"],
        model_seed=config["seed"],
        **config["collect"],
    )


if __name__ == "__main__":
    main()
