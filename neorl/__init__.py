import importlib


def make(dataset_name: str, reward_func=None, done_func=None):
    try:
        task, version, _ = dataset_name.split("-")
        task = task + "-" + version
        if task in ["HalfCheetah-v3", "Walker2d-v3", "Hopper-v3"]:
            from neorl.neorl_envs import mujoco

            env = mujoco.make_env(task)
        else:
            raise ValueError(f"Env {task} is not supported!")
    except Exception as e:
        print(f"Warning: Env {task} can not be create. Pleace Check!")
        raise e
    env.reset()
    env.set_name(dataset_name)

    try:
        default_reward_func = importlib.import_module(
            f"neorl.neorl_envs.{task}.{task}_reward"
        ).get_reward
    except ModuleNotFoundError:
        default_reward_func = None

    env.set_reward_func(default_reward_func if reward_func is None else reward_func)

    try:
        default_done_func = importlib.import_module(
            f"neorl.neorl_envs.{task}.{task}_done"
        ).get_done
    except ModuleNotFoundError:
        default_done_func = None

    env.set_done_func(default_done_func if done_func is None else done_func)

    return env
