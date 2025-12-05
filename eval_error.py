from jax import random
import jax.numpy as jnp
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from offline_world.cont_ensemble import LearnedContEnv
from experience.wrapper import make_env
from experience.world_buffer import get_history_dataset

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd


def fix_numerical_issue(data):
    data[np.isposinf(data)] = np.finfo(np.float32).max
    data[np.isneginf(data)] = np.finfo(np.float32).min
    # forward last non-nan value
    data = pd.DataFrame(data).ffill(axis=0).to_numpy()
    return data


sns.set_theme(context="talk", style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 16,  # Default font size for text
        "axes.titlesize": 17,  # Font size for axes titles
        "axes.labelsize": 16,  # Font size for axes labels
        "xtick.labelsize": 16,  # Font size for x-tick labels
        "ytick.labelsize": 16,  # Font size for y-tick labels
        "legend.fontsize": 16,  # Font size for legend
        "figure.titlesize": 17,  # Font size for figure title
        "axes.formatter.useoffset": False,
        "axes.formatter.offset_threshold": 1,
    }
)
colors = ["green", "gray", "orange", "red"]


class CompoundingErrorEvaluator:
    """
    Similar to the Collector, but for evaluating compounding error over a given dataset.
    This dataset is composed of (initial state, action sequence) pairs from
    - in-distribution training data
    - out-of-distribution data
    """

    def __init__(self, envs):
        self.envs = envs

    def __call__(
        self,
        eval_dataset,  # dict of numpy arrays of shape (T_max, N, dim)
        eval_name: str,
        train_name: str,
        setting: str,
        key: random.PRNGKey,
    ):

        # 1. reset the initial states of dim (N, S) and other infos in train_envs
        self.envs.states = eval_dataset["observation"][
            0, :, : self.envs.state_dim
        ].copy()
        self.envs.timesteps = np.zeros((self.envs.states.shape[0],), dtype=np.int32)
        self.envs.truncateds = np.zeros((self.envs.states.shape[0],), dtype=bool)
        self.envs.terminateds = np.zeros((self.envs.states.shape[0],), dtype=bool)

        # 2. rollout the action sequence in eval dataset using the world model
        next_rewards = []  # (r1, r2, ..., rT)
        uncertainties = []  # (u1, u2, ..., uT)
        observations = []  # (o1, ..., oT) where oT is last observation

        ## 2. iterate through dataset actions (NOTE: last action is dummy)
        T_max = eval_dataset["action"].shape[0]
        for plan_step in tqdm(range(T_max), desc="Compounding eval"):
            action = eval_dataset["action"][plan_step].copy()  # (N, A)

            key, step_key = random.split(key, 2)
            # we disable terminated and truncated signals for evaluation
            (observation, reward, _, _, uncertainty) = self.envs.step(
                action, key=step_key
            )

            next_rewards.append(reward)
            uncertainties.append(uncertainty)
            observations.append(observation)

        # convert to numpy arrays of shape (T_max, N, *)
        next_rewards = np.asarray(jnp.array(next_rewards), dtype=np.float64)
        uncertainties = np.asarray(jnp.array(uncertainties), dtype=np.float64)
        observations = np.asarray(jnp.array(observations), dtype=np.float64)

        dones = np.logical_or(
            eval_dataset["next_terminated"], eval_dataset["next_truncated"]
        )
        first_done_idx = np.argmax(dones, axis=0)  # (N,)
        h_max = np.max(first_done_idx) + 1
        mask = (
            np.arange(h_max)[:, None] > first_done_idx[None, :]
        )  # in fact, a shift of dones

        ## plot
        plot_dir = os.path.join(self.envs.plot_dir, eval_name, setting)
        os.makedirs(plot_dir, exist_ok=True)
        num_plots = 4
        fig, axes = plt.subplots(
            1, num_plots, figsize=(4 * num_plots, 4), constrained_layout=True
        )
        axes = iter(axes.flatten())

        ########### Compound State Error ###########
        delta = (observations[:-1] - eval_dataset["observation"][1:])[
            :h_max, :, : self.envs.state_dim
        ]
        compound_state_error = np.sqrt(np.nanmean(delta**2, axis=-1))  # (H, N)
        compound_state_error = fix_numerical_issue(compound_state_error)
        compound_state_error[mask] = np.nan
        np.save(
            os.path.join(plot_dir, "compound_state_error.npy"), compound_state_error
        )

        ax = next(axes)
        med = np.nanmedian(compound_state_error, axis=1)
        x = np.arange(1, med.shape[0] + 1)  # start from 1
        ax.plot(x, med, color="black", linewidth=2, label="Median", zorder=3)
        ax.plot(x, compound_state_error, alpha=0.8, lw=1, zorder=1, rasterized=True)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        compound_ylim = ax.get_ylim()
        ax.set_xlabel(r"Planning Step $t$")
        ax.set_title(r"Compounding Error: $\mathrm{RMSE}(\hat{s}_t, s_t)$")

        ########### Compound State Norm ###########
        compound_state_norm = observations[:-1][:h_max, :, : self.envs.state_dim]
        compound_state_norm = np.sqrt(np.nanmean(compound_state_norm**2, axis=-1))
        compound_state_norm = fix_numerical_issue(compound_state_norm)
        compound_state_norm[mask] = np.nan
        np.save(os.path.join(plot_dir, "compound_state_norm.npy"), compound_state_norm)

        ax = next(axes)
        med = np.nanmedian(compound_state_norm, axis=1)
        ax.plot(x, med, color="black", linewidth=2, label="Median", zorder=3)
        ax.plot(x, compound_state_norm, alpha=0.8, lw=1, zorder=1, rasterized=True)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_ylim(compound_ylim)
        ax.set_xlabel(r"Planning Step $t$")
        ax.set_title(r"Predicted Norm: $\mathrm{RMS}(\hat{s}_t)$")

        ########### Compound Reward Bias ###########
        compound_reward_bias = (next_rewards - eval_dataset["next_reward"])[:h_max]
        compound_reward_bias = fix_numerical_issue(compound_reward_bias)
        compound_reward_bias[mask] = np.nan
        np.save(
            os.path.join(plot_dir, "compound_reward_bias.npy"), compound_reward_bias
        )

        ax = next(axes)
        med = np.nanmedian(compound_reward_bias, axis=1)
        ax.plot(x, med, color="black", linewidth=2, label="Median", zorder=3)
        ax.plot(x, compound_reward_bias, alpha=0.8, lw=1, zorder=1, rasterized=True)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel(r"Planning Step $t$")
        ax.set_title(r"Compound Reward Bias: $\hat{r}_t - r_t$")

        ########### Scatter ###########
        x = uncertainties[:-1].ravel()
        y = compound_state_error.ravel()
        np.save(os.path.join(plot_dir, "scatter.npy"), np.stack([x, y], axis=1))
        np.save(os.path.join(plot_dir, "quantile.npy"), self.envs.common_quantiles)

        valid = np.logical_and(~np.isnan(x), ~np.isnan(y))
        rho, _ = spearmanr(x[valid], y[valid])

        ax = next(axes)
        ax.scatter(x, y, s=3, alpha=0.2, rasterized=True)
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_ylim(compound_ylim)
        ax.set_xlabel("Estimated Uncertainty")
        ax.set_ylabel("Compounding Error")
        ax.set_title(f"Spearman rank coef = {rho:.2f}")
        y_positions = [0.68, 0.78, 0.88, 0.98]
        for i, (label, xval) in enumerate(self.envs.common_quantiles.items()):
            color = colors[i % len(colors)]
            ax.axvline(x=xval, color=color, ls="--", lw=2)
            # place the label near the right edge, vertically centered on the line
            ax.text(
                xval,
                y_positions[i % len(y_positions)],
                label,
                color=color,
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor="white", alpha=0.7),
            )

        fig.suptitle(
            f"Trained on {train_name}; Evaluated on {eval_name}; Setting: {setting}",
            fontweight="bold",
        )
        plt.savefig(os.path.join(plot_dir, "result.pdf"), dpi=200, bbox_inches="tight")
        plt.close()
        print("save plot to", plot_dir)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # Convert to dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)
    os.environ["WANDB_MODE"] = "disabled"
    key = random.PRNGKey(config["seed"] + 10)
    N = 200  # num of trajectories to eval; hc-med-rep has 202 trajs in total

    """
    Build eval dataset
    """
    eval_dataset_name = (
        config["dataset_name"].split("-")[0] + "-" + config["eval_dataset"]
    )
    eval_env = make_env(config["domain"], eval_dataset_name)
    eval_dataset = get_history_dataset(config["domain"], eval_env)
    print("Evaluating on dataset:", eval_dataset_name)
    print(f"Downsample {N} of {len(eval_dataset)} trajectories")

    # subsample N
    rng = np.random.default_rng(config["seed"] + 20)
    idx = rng.choice(len(eval_dataset), size=N, replace=False)
    eval_dataset = [eval_dataset[i] for i in idx]

    ## right-pad the dataset into fixed-length sequences
    T_max = max([d["observation"].shape[0] for d in eval_dataset])
    keys = list(eval_dataset[0].keys())
    out = {}
    first = eval_dataset[0]
    for k in keys:
        arr0 = first[k]
        if arr0.ndim == 1:
            out[k] = np.empty((T_max, N), dtype=arr0.dtype)
        else:
            out[k] = np.empty((T_max, N, arr0.shape[1]), dtype=arr0.dtype)

    # fill with data + right padding (repeat last value)
    for i, d in enumerate(eval_dataset):
        T = d[keys[0]].shape[0]
        pad_len = T_max - T

        for k in keys:
            arr = d[k]
            out[k][:T, i] = arr
            if pad_len > 0:
                out[k][T:, i] = arr[-1]  # broadcast last value/row

    """
    Trained world ensemble
    """
    train_envs = LearnedContEnv()
    train_envs.load_model_and_data(
        domain=config["domain"],
        dataset_name=config["dataset_name"],
        save_dir=config["ensemble"]["save_dir"],
        plot_dir="plt/d4rl_loco",  # overwrite as we have LN folder
        model_seed=config["seed"],
        **config["collect"],
    )

    ## Evaluator
    evaluator = CompoundingErrorEvaluator(train_envs)
    setting = "with_LN" if cfg["ensemble"]["has_ln"] else "without_LN"
    evaluator(
        eval_dataset=out,
        eval_name=eval_dataset_name,
        train_name=config["dataset_name"],
        setting=setting,
        key=key,
    )


if __name__ == "__main__":
    main()
