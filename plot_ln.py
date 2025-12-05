import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dataset",
    type=str,
    required=True,
)
parser.add_argument(
    "--eval_dataset",
    type=str,
    required=True,
)
args = parser.parse_args()

train_dataset = args.train_dataset
eval_dataset = train_dataset.split("-")[0] + "-" + args.eval_dataset

plot_dir = f"plt/d4rl_loco/{train_dataset}/ensemble_seed0/{eval_dataset}"
no_ln_dir = os.path.join(plot_dir, "without_LN")
with_ln_dir = os.path.join(plot_dir, "with_LN")

save_dir = f"plt/compound/{train_dataset}/{eval_dataset}"
os.makedirs(save_dir, exist_ok=True)
plot_path = os.path.join(save_dir, "main.pdf")

sns.set_theme(context="talk", style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 17,  # Default font size for text
        "axes.titlesize": 19,  # Font size for axes titles
        "axes.labelsize": 17,  # Font size for axes labels
        "xtick.labelsize": 17,  # Font size for x-tick labels
        "ytick.labelsize": 17,  # Font size for y-tick labels
        "legend.fontsize": 17,  # Font size for legend
        "figure.titlesize": 20,  # Font size for figure title
        "axes.formatter.useoffset": False,
        "axes.formatter.offset_threshold": 1,
    }
)
colors = ["green", "gray", "orange", "red"]
num_plots = 4
fig, axes = plt.subplots(
    1, num_plots, figsize=(5 * num_plots, 4), constrained_layout=True
)
axes = iter(axes.flatten())


def set_x_axis(ax, x_max):
    k_max = int(np.ceil(np.log2(x_max)))
    ticks = 2 ** np.arange(0, k_max + 1)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xlim(1, x_max)
    ax.set_xticklabels([rf"$2^{{{k}}}$" for k in range(0, k_max + 1)])


def set_y_axis(ax, y_max):
    ax.set_yscale("log", base=10)
    ax.set_ylim(0.01, min(1e6, y_max))


def set_reward_axis(ax):
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-1e3, 1e3)


def plot_with_percentiles(
    ax,
    data,
    color,
    label,
    p_low=10,
    p_high=90,
    fill_alpha=0.2,
):
    # forward fill NaN values
    data = pd.DataFrame(data).ffill(axis=0).to_numpy()

    p_low_val = np.nanpercentile(data, p_low, axis=1)
    p_med_val = np.nanmedian(data, axis=1)
    p_high_val = np.nanpercentile(data, p_high, axis=1)

    x = np.arange(1, p_med_val.shape[0] + 1)  # start from 1
    ax.plot(x, p_med_val, color=color, linewidth=2, label=label, alpha=0.8)
    ax.fill_between(
        x,
        p_low_val,
        p_high_val,
        color=color,
        alpha=fill_alpha,
        linewidth=0,
        label="_nolegend_",
    )
    return p_low_val, p_med_val, p_high_val


########### Compound State Error ###########
ax = next(axes)
compound_state_error_no_ln = np.load(
    os.path.join(no_ln_dir, "compound_state_error.npy")
)
compound_state_error_with_ln = np.load(
    os.path.join(with_ln_dir, "compound_state_error.npy")
)

plot_with_percentiles(
    ax,
    compound_state_error_no_ln,
    color="red",
    label="No LayerNorm",
    p_low=5,
    p_high=95,
)
plot_with_percentiles(
    ax,
    compound_state_error_with_ln,
    color="blue",
    label="With LayerNorm (ours)",
    p_low=5,
    p_high=95,
)

ax.set_xlabel(r"Planning Step $t$")
set_x_axis(ax, compound_state_error_no_ln.shape[0])
set_y_axis(
    ax,
    max(np.nanmax(compound_state_error_no_ln), np.nanmax(compound_state_error_with_ln)),
)
_, y_max = ax.get_ylim()
ax.set_title(r"Compounding Error: $\mathrm{RMSE}(\hat{s}_t, s_t)$")

########### Compound State Norm ###########
ax = next(axes)
compound_state_norm_no_ln = np.load(os.path.join(no_ln_dir, "compound_state_norm.npy"))
compound_state_norm_with_ln = np.load(
    os.path.join(with_ln_dir, "compound_state_norm.npy")
)
plot_with_percentiles(
    ax, compound_state_norm_no_ln, color="red", label="No LayerNorm", p_low=5, p_high=95
)
plot_with_percentiles(
    ax,
    compound_state_norm_with_ln,
    color="blue",
    label="With LayerNorm (ours)",
    p_low=5,
    p_high=95,
)

ax.set_xlabel(r"Planning Step $t$")
set_x_axis(ax, compound_state_error_no_ln.shape[0])
set_y_axis(ax, y_max)
ax.tick_params(labelleft=False)
ax.set_ylabel("")
ax.set_title(r"Predicted Norm: $\mathrm{RMS}(\hat{s}_t)$")
ax.legend(framealpha=0.5)

########### Compound Reward Bias ###########
ax = next(axes)
compound_reward_bias_no_ln = np.load(
    os.path.join(no_ln_dir, "compound_reward_bias.npy")
)
compound_reward_bias_with_ln = np.load(
    os.path.join(with_ln_dir, "compound_reward_bias.npy")
)
plot_with_percentiles(
    ax,
    compound_reward_bias_no_ln,
    color="red",
    label="No LayerNorm",
    p_low=5,
    p_high=95,
)
plot_with_percentiles(
    ax,
    compound_reward_bias_with_ln,
    color="blue",
    label="With LayerNorm (ours)",
    p_low=5,
    p_high=95,
)

ax.set_xlabel(r"Planning Step $t$")
set_x_axis(ax, compound_state_error_no_ln.shape[0])
set_reward_axis(ax)
ax.set_title(r"Compound Reward Bias: $\hat{r}_t - r_t$")

########### Scatter ###########
ax = next(axes)

xy = np.load(os.path.join(with_ln_dir, "scatter.npy"))
x, y = xy[:, 0], xy[:, 1]
valid = np.logical_and(~np.isnan(x), ~np.isnan(y))
rho, _ = spearmanr(x[valid], y[valid])

ax.scatter(x, y, s=3, alpha=0.2, rasterized=True)

ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)
ax.set_xlabel("Estimated Uncertainty")
ax.set_ylabel("Compounding Error")
ax.set_title(r"$\bf{With\ LayerNorm:}$" + f"coef = {rho:.2f}")
y_positions = [0.68, 0.78, 0.88, 0.98]
common_quantiles = np.load(
    os.path.join(with_ln_dir, "quantile.npy"), allow_pickle="TRUE"
).item()

for i, (label, xval) in enumerate(common_quantiles.items()):
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
    f"Trained on {train_dataset}; Evaluated on {eval_dataset}", fontweight="bold"
)
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print("save plot to", plot_path)
