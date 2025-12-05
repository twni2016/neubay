import gym, d4rl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import wandb
import os

wall_coords = None
wall_scale = None


def setup_maze(dataset_name):
    global wall_coords, wall_scale
    bg_env = gym.make(dataset_name)
    # Access internal maze map (1=wall, 0=empty)
    maze_map = bg_env.spec.kwargs["maze_map"]
    wall_scale = bg_env.spec.kwargs["maze_size_scaling"]

    wall_coords = []
    for row in range(len(maze_map)):
        for col in range(len(maze_map[0])):
            if maze_map[row][col] == 1:
                x, y = bg_env.unwrapped._rowcol_to_xy((row, col))
                wall_coords.append((x, y))
    bg_env.close()


def plot_trajs(real_rollouts, eval_times, save_fig=False):
    # Plot trajectories for each eval environment
    fig, ax = plt.subplots(dpi=200)
    xs, ys = zip(*wall_coords)

    for x, y in wall_coords:
        ax.add_patch(
            Rectangle(
                (x - 0.5 * wall_scale, y - 0.5 * wall_scale),
                wall_scale,
                wall_scale,
                facecolor="black",
                edgecolor="black",
            )
        )
    ax.set_xlim(min(xs) - 0.5 * wall_scale, max(xs) + 0.5 * wall_scale)
    ax.set_ylim(min(ys) - 0.5 * wall_scale, max(ys) + 0.5 * wall_scale)

    # Plot trajectories
    for idx, traj in enumerate(real_rollouts):
        obs = traj["observation"]
        x = obs[:, 0]
        y = obs[:, 1]
        plt.plot(x, y, color="red", alpha=0.3)
        plt.scatter(x[-1], y[-1], color="blue", marker="x", s=1, zorder=3)

    plt.title(f"Eval OBS {eval_times} Trajectories Bundle")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    img = wandb.Image(plt.gcf())
    if save_fig:
        eval_plot_dir = os.path.join("plots", "eval_trajectories")
        os.makedirs(eval_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(eval_plot_dir, f"{eval_times}.png"))
    plt.close()
    return img
