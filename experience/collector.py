import numpy as np
from jax import random
import jax.numpy as jnp
from online_rl.common import feature_metrics
import os, time
from tqdm import tqdm


class BanditCollector:
    """
    This is a simplified version of the collector, designed for fixed-horizon bandit problems.

    Rolling out the agent in a vectorized **learned** environment (i.e., ensemble world model)
    Start from t = 0 and end at t = T (fixed horizon), a greatly simplified version
        since planning in bandit does not suffer from compounding error
    """

    def __init__(self, envs):
        # vectorized envs
        self.envs = envs

    def __call__(self, agent, parallel_size: int, report: bool, key: random.PRNGKey):
        starts = []  # (s0, s1, ..., sT) where s0 = True, the rest = False
        observations = []  # (o0, o1, ..., oT) where oT is last observation
        actions = []  # (a0, a1, ..., aT) where aT is dummy action
        next_rewards = []  # (r1, r2, ..., rT+1) where rT+1 is dummy reward
        terminateds = []  # (d1, d2, ..., dT+1) where dT+1 is dummy terminated
        truncateds = []  # (d1, d2, ..., dT+1) where dT+1 is dummy terminated

        key, reset_key = random.split(key)
        observation = self.envs.reset(parallel_size=parallel_size, key=reset_key)
        done = False
        start = np.array([True] * parallel_size)

        starts.append(start)
        observations.append(observation)  # (N, dim)

        # get initial states: L[(N, 1, d_hidden)] complex
        recurrent_state = agent.initial_state(parallel_size)
        if report:  # report the plasticity metrics on features
            features = []

        while not done:
            key, action_key, step_key = random.split(key, 3)
            action, recurrent_state, feature = agent(
                x=jnp.expand_dims(observation, 1),  # (N, 1, dim)
                state=recurrent_state,
                start=jnp.expand_dims(start, 1),  # (N, 1)
                key=action_key,
            )  # action: # (N, dim)
            (observation, reward, terminated, truncated) = self.envs.step(
                action, key=step_key
            )
            start = np.array([False] * parallel_size)

            actions.append(action)
            next_rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)

            starts.append(start)
            observations.append(observation)

            if report:
                features.append(feature)

            done = terminated.all() or truncated.all()

        rollout_reward = np.array(next_rewards).sum(axis=0).mean()
        data_size = parallel_size * len(next_rewards)
        metrics = {"reward": rollout_reward, "data_size": data_size}

        if report:
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
        rollout_data = [
            {k: v[:, n] for k, v in data.items()} for n in range(parallel_size)
        ]
        return rollout_data, metrics


class ContCollector:
    def __init__(self, envs):
        # vectorized envs
        self.envs = envs

    def __call__(
        self,
        agent,
        deterministic: bool,
        start_from_s0: bool,
        parallel_size: int,
        key: random.PRNGKey,
        visualize: bool = False,
        **kwargs,
    ):

        # 1. warm up the agent with history
        ## NOTE: we may sample finished trajs as histories if start_from_s0 is False
        key, reset_key, warmup_key, step_key = random.split(key, 4)
        recurrent_state = agent.initial_state(parallel_size)
        histories, history_infos = self.envs.reset(
            start_from_s0=start_from_s0, parallel_size=parallel_size, key=reset_key
        )

        action, recurrent_state, _ = agent(
            x=history_infos["padded_obs"],  # (N, T, dim)
            state=recurrent_state,
            start=jnp.array(history_infos["padded_start"]),  # (N, T)
            key=warmup_key,
            deterministic=deterministic,
        )

        # 2. rollout the agent in world model
        actions = []  # (a0, a1, ..., aT-1) where aT-1 is last action
        next_rewards = []  # (r1, r2, ..., rT)
        uncertainties = []  # (u1, u2, ..., uT)
        starts = []  # (s1, ..., sT)
        observations = []  # (o1, ..., oT) where oT is last observation
        terminateds = []  # (d1, ..., dT)
        truncateds = []  # (d1, ..., dT)

        for plan_step in range(self.envs.max_rollout_len):
            if plan_step > 0:
                key, action_key, step_key = random.split(key, 3)
                action, recurrent_state, _ = agent(
                    x=jnp.expand_dims(observation, 1),  # (N, 1, dim)
                    state=recurrent_state,
                    start=jnp.expand_dims(start, 1),  # (N, 1)
                    key=action_key,
                    deterministic=deterministic,
                )

            (observation, reward, terminated, truncated, uncertainty) = self.envs.step(
                action, key=step_key
            )
            start = np.array([False] * parallel_size)
            if plan_step == self.envs.max_rollout_len - 1:
                # overwrite all truncation
                truncated = np.ones_like(truncated, dtype=bool)

            actions.append(action)
            next_rewards.append(reward)
            uncertainties.append(uncertainty)
            starts.append(start)
            observations.append(observation)
            terminateds.append(terminated)
            truncateds.append(truncated)

            ## check if all rollouts are done
            done = np.logical_or(terminated, truncated).all()
            if done:
                break

        # convert to numpy arrays of shape (H, N, *)
        actions = np.asarray(jnp.array(actions))
        next_rewards = np.asarray(jnp.array(next_rewards))
        uncertainties = np.asarray(jnp.array(uncertainties))
        starts = np.array(starts)
        observations = np.asarray(jnp.array(observations))
        terminateds = np.array(terminateds)
        truncateds = np.array(truncateds)

        # 3. extract the valid rollouts
        dones = np.logical_or(
            np.concatenate([history_infos["terminateds"][None], terminateds], axis=0),
            np.concatenate([history_infos["truncateds"][None], truncateds], axis=0),
        )  # (H+1, N)
        first_done_idx = np.argmax(
            dones, axis=0
        )  # (N,), within {0, ..., H}, i.e., rollout lengths

        rollouts = []
        for batch_idx, done_idx in enumerate(first_done_idx):
            rollout = {
                "start": starts[:done_idx, batch_idx],
                "observation": observations[:done_idx, batch_idx],
                "action": actions[:done_idx, batch_idx],
                "next_reward": next_rewards[:done_idx, batch_idx],
                "next_terminated": terminateds[:done_idx, batch_idx],
                "next_truncated": truncateds[:done_idx, batch_idx],
                "uncertainty": uncertainties[:done_idx, batch_idx],
            }
            rollouts.append(rollout)

        if visualize:
            # for debugging, visualize the rollouts
            self.visualize(
                history_infos, rollouts, uncertainties, first_done_idx, **kwargs
            )

        # 4. concatenate history with rollout
        data = []
        for history, rollout in zip(histories, rollouts):
            traj = {
                key: np.concatenate([history[key], rollout[key]], axis=0)
                for key in [
                    "start",
                    "observation",
                    "action",
                    "next_reward",
                    "next_terminated",
                    "next_truncated",
                ]
            }
            # distinguish history from rollout with real flags
            traj["real"] = np.zeros_like(traj["next_terminated"], dtype=bool)
            traj["real"][: len(history["next_terminated"])] = True
            # add dummy last transition
            for key in [
                "action",
                "next_reward",
                "next_terminated",
                "next_truncated",
                "real",
            ]:
                traj[key] = np.concatenate([traj[key], traj[key][-1:]], axis=0)
            data.append(traj)

        # 5. report metrics
        returns = [np.sum(traj["next_reward"][:-1]) for traj in data]
        lengths = [len(traj["next_reward"]) for traj in data]
        metrics = {
            "reward": np.mean(returns),
            "reward_std": np.std(returns),
            "length": np.mean(lengths) - 1,
            "length_std": np.std(lengths),
            "horizon_mean": np.mean(first_done_idx),
            "data_size": np.sum(lengths) - parallel_size,
        }
        for q in [0.25, 0.5, 0.75, 1.0]:  # min is always 0 if we have finished history
            metrics[f"horizon_quantile{q}"] = np.quantile(first_done_idx, q)

        return data, metrics

    def call_markov(
        self,
        agent,
        deterministic: bool,
        start_from_s0: bool,
        parallel_size: int,
        key: random.PRNGKey,
        **kwargs,
    ):
        """
        __call__ for Markov agents
        """
        # 1. "warm up" the markov agent
        key, reset_key, warmup_key, step_key = random.split(key, 4)
        histories, history_infos = self.envs.reset(
            start_from_s0=start_from_s0, parallel_size=parallel_size, key=reset_key
        )
        ## markov agent only use the last observation
        observation = history_infos["padded_obs"][:, -1, : self.envs.state_dim]

        action = agent(
            x=observation,  # (N, dim)
            key=warmup_key,
            eval_mode=True,
            deterministic=deterministic,
        )

        # 2. rollout the agent in world model
        actions = []  # (a0, a1, ..., aT-1) where aT-1 is last action
        next_rewards = []  # (r1, r2, ..., rT)
        uncertainties = []  # (u1, u2, ..., uT)
        observations = []  # (o1, ..., oT) where oT is last observation
        terminateds = []  # (d1, ..., dT)
        truncateds = []  # (d1, ..., dT)

        for plan_step in range(self.envs.max_rollout_len):
            if plan_step > 0:
                key, action_key, step_key = random.split(key, 3)
                action = agent(
                    x=observation,  # (N, dim)
                    key=action_key,
                    eval_mode=True,
                    deterministic=deterministic,
                )

            (observation, reward, terminated, truncated, uncertainty) = self.envs.step(
                action, key=step_key
            )
            observation = observation[:, : self.envs.state_dim]
            if plan_step == self.envs.max_rollout_len - 1:
                # overwrite all truncation
                truncated = np.ones_like(truncated, dtype=bool)

            actions.append(action)
            next_rewards.append(reward)
            uncertainties.append(uncertainty)
            observations.append(observation)
            terminateds.append(terminated)
            truncateds.append(truncated)

            ## check if all rollouts are done
            done = np.logical_or(terminated, truncated).all()
            if done:
                break

        # convert to numpy arrays of shape (H, N, *)
        actions = np.asarray(jnp.array(actions))
        next_rewards = np.asarray(jnp.array(next_rewards))
        uncertainties = np.asarray(jnp.array(uncertainties))
        observations = np.asarray(jnp.array(observations))
        terminateds = np.array(terminateds)
        truncateds = np.array(truncateds)

        # 3. extract the valid rollouts
        dones = np.logical_or(
            np.concatenate([history_infos["terminateds"][None], terminateds], axis=0),
            np.concatenate([history_infos["truncateds"][None], truncateds], axis=0),
        )  # (H+1, N)
        first_done_idx = np.argmax(
            dones, axis=0
        )  # (N,), within {0, ..., H}, i.e., rollout lengths

        rollouts = []
        for batch_idx, done_idx in enumerate(first_done_idx):
            rollout = {
                "observation": observations[:done_idx, batch_idx],
                "action": actions[:done_idx, batch_idx],
                "next_reward": next_rewards[:done_idx, batch_idx],
                "next_terminated": terminateds[:done_idx, batch_idx],
                "next_truncated": truncateds[:done_idx, batch_idx],
                "uncertainty": uncertainties[:done_idx, batch_idx],
            }
            rollouts.append(rollout)

        # 4. concatenate history with rollout
        data = []
        for history, rollout in zip(histories, rollouts):
            traj = {
                key: np.concatenate(
                    [
                        (
                            history[key][:, : self.envs.state_dim]
                            if key == "observation"
                            else history[key]
                        ),
                        rollout[key],
                    ],
                    axis=0,
                )
                for key in [
                    "observation",
                    "action",
                    "next_reward",
                    "next_terminated",
                    "next_truncated",
                ]
            }
            # distinguish history from rollout with real flags
            traj["real"] = np.zeros_like(traj["next_terminated"], dtype=bool)
            traj["real"][: len(history["next_terminated"])] = True

            # transform to markov transition style
            traj["next_observation"] = traj["observation"][1:].copy()
            traj["observation"] = traj["observation"][:-1].copy()

            data.append(traj)

        # 5. report metrics
        returns = [np.sum(traj["next_reward"][:-1]) for traj in data]
        lengths = [len(traj["next_reward"]) for traj in data]
        metrics = {
            "reward": np.mean(returns),
            "reward_std": np.std(returns),
            "length": np.mean(lengths) - 1,
            "length_std": np.std(lengths),
            "horizon_mean": np.mean(first_done_idx),
            "data_size": np.sum(lengths) - parallel_size,
        }
        for q in [0.25, 0.5, 0.75, 1.0]:  # min is always 0 if we have finished history
            metrics[f"horizon_quantile{q}"] = np.quantile(first_done_idx, q)

        return data, metrics

    def visualize(
        self,
        history_infos,
        rollouts,
        uncertainties,
        first_done_idx,
        title: str,
        filename: str,
    ):
        """
        This function is only used for debugging purposes.
        The compounding error plots in the paper are generated by `eval_error.py`, not this one.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import spearmanr
        import pandas as pd

        def forward_fill(arr):
            # forward last non-nan values to fill nans
            return pd.DataFrame(arr).ffill(axis=0).to_numpy()

        mask = np.arange(len(uncertainties))[:, None] < first_done_idx
        masked_unc = np.where(mask, uncertainties, np.nan)  # (H, N)

        compound_state_error = np.full(masked_unc.shape, np.nan)  # \|\hat s_t - s_t\|_2
        compound_state_norm = np.full(masked_unc.shape, np.nan)  # \|\hat s_t\|_2
        compound_reward_bias = np.full(masked_unc.shape, np.nan)  # \hat r_t - r_t

        for batch_idx, rollout in enumerate(rollouts):
            if len(rollout["action"]) == 0:  # finished history
                continue
            self.envs.oracle_env.reset(
                history_infos["padded_obs"][batch_idx, -1, : self.envs.state_dim]
            )
            for step_idx, (action, pred_next_obs, pred_next_reward) in enumerate(
                zip(rollout["action"], rollout["observation"], rollout["next_reward"])
            ):
                pred_next_state = pred_next_obs[: self.envs.state_dim]
                next_state, next_reward = self.envs.oracle_env.step(action)

                # cast to float64 for overflow safety
                next_state = np.float64(next_state)
                next_reward = np.float64(next_reward)
                pred_next_state = np.float64(pred_next_state)
                pred_next_reward = np.float64(pred_next_reward)

                compound_state_norm[step_idx, batch_idx] = np.sqrt(
                    np.nanmean(pred_next_state**2)
                )
                compound_state_error[step_idx, batch_idx] = np.sqrt(
                    np.nanmean((next_state - pred_next_state) ** 2)
                )
                compound_reward_bias[step_idx, batch_idx] = (
                    pred_next_reward - next_reward
                )

        ## plot
        os.makedirs(self.envs.plot_dir, exist_ok=True)
        plot_path = os.path.join(self.envs.plot_dir, filename)
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
        num_plots = 4
        fig, axes = plt.subplots(
            1, num_plots, figsize=(4 * num_plots, 4), constrained_layout=True
        )
        axes = iter(axes.flatten())

        ########### Compound State Error ###########
        ax = next(axes)
        med = np.nanmedian(forward_fill(compound_state_error), axis=1)
        x = np.arange(1, med.shape[0] + 1)  # start from 1
        ax.plot(x, med, color="black", linewidth=2, label="Median", zorder=3)
        ax.plot(x, compound_state_error, alpha=0.8, lw=1, zorder=1, rasterized=True)
        np.save(
            plot_path.replace(".pdf", "_compound_state_error.npy"), compound_state_error
        )

        ax.set_xlabel(r"Planning Step $t$")
        if len(x) > 200:
            ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        compound_ylim = ax.get_ylim()
        ax.set_title(r"Compounding Error: $\mathrm{RMSE}(\hat{s}_t, s_t)$")
        ax.legend(framealpha=0.5)

        ########### Compound State Norm ###########
        ax = next(axes)
        med = np.nanmedian(forward_fill(compound_state_norm), axis=1)
        ax.plot(x, med, color="black", linewidth=2, label="Median", zorder=3)
        ax.plot(x, compound_state_norm, alpha=0.8, lw=1, zorder=1, rasterized=True)
        np.save(
            plot_path.replace(".pdf", "_compound_state_norm.npy"), compound_state_norm
        )

        ax.set_xlabel(r"Planning Step $t$")
        if len(x) > 200:
            ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_ylim(compound_ylim)
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")
        ax.set_title(r"Predicted Norm: $\mathrm{RMS}(\hat{s}_t)$")

        ########### Compound Reward Bias ###########
        ax = next(axes)
        med = np.nanmedian(forward_fill(compound_reward_bias), axis=1)
        ax.plot(x, med, color="black", linewidth=2, label="Median", zorder=3)
        ax.plot(x, compound_reward_bias, alpha=0.8, lw=1, zorder=1, rasterized=True)

        ax.set_xlabel(r"Planning Step $t$")
        if len(x) > 200:
            ax.set_xscale("log", base=10)
        ax.set_title(r"Compound Reward Bias: $\hat{r}_t - r_t$")
        np.save(
            plot_path.replace(".pdf", "_compound_reward_bias.npy"),
            compound_reward_bias,
        )

        ########### Scatter ###########
        ax = next(axes)

        x = masked_unc.ravel()
        y = compound_state_error.ravel()
        valid = np.logical_and(~np.isnan(x), ~np.isnan(y))
        rho, _ = spearmanr(x[valid], y[valid])

        ax.scatter(x, y, s=3, alpha=0.2, rasterized=True)
        np.save(plot_path.replace(".pdf", "_scatter.npy"), np.stack([x, y], axis=1))
        np.save(plot_path.replace(".pdf", "_quantile.npy"), self.envs.common_quantiles)

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

        fig.suptitle(self.envs.dataset_name + ": " + title, fontweight="bold")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("save plot to", plot_path)
