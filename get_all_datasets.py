import gym, d4rl

# https://github.com/Farama-Foundation/D4RL/wiki/Tasks
dataset_names = [
    # mujoco domain: all have next_observations
    # "halfcheetah-random-v2",
    # "halfcheetah-medium-v2",
    # "halfcheetah-medium-replay-v2",
    # "halfcheetah-medium-expert-v2",
    # "hopper-random-v2",
    # "hopper-medium-v2",
    # "hopper-medium-replay-v2",
    # "hopper-medium-expert-v2",
    # "walker2d-random-v2",
    # "walker2d-medium-v2",
    # "walker2d-medium-replay-v2",
    # "walker2d-medium-expert-v2",
    # antmaze domain: all **do not** have next_observations, used in LEQ
    # v2 and v0 has same datasets, but v2 evaluation is more stable: https://github.com/Farama-Foundation/D4RL/pull/128
    # "antmaze-umaze-v2",
    # "antmaze-umaze-diverse-v2",
    # "antmaze-medium-play-v2",
    # "antmaze-medium-diverse-v2",
    # "antmaze-large-play-v2",
    # "antmaze-large-diverse-v2",
    # adroit domain: all **do not** have next_observations, used in CORL and MOBILE
    # v1 human demonstrations may time out earlier.
    # "pen-human-v1",
    # "pen-cloned-v1",
    # "door-human-v1",
    # "door-cloned-v1",
    # "hammer-human-v1",
    # "hammer-cloned-v1",
    # neorl domain: all have next_observations, but not timeouts
    # "HalfCheetah-v3-low",
    # "HalfCheetah-v3-medium",
    # "HalfCheetah-v3-high",
    # "Hopper-v3-low",
    # "Hopper-v3-medium",
    # "Hopper-v3-high",
    # "Walker2d-v3-low",
    # "Walker2d-v3-medium",
    # "Walker2d-v3-high",
]

for dataset_name in dataset_names:
    print(f"Loading dataset: {dataset_name}")
    try:
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
    except:
        import neorl

        env = neorl.make(dataset_name)
        # for neorl, we follow MOBILE and LEQ that only uses the train split
        dataset, _ = env.get_dataset(train_num=1000, need_val=False)
        dataset["rewards"] = dataset["reward"]

    print(f"{env.observation_space.shape =}, {env.action_space.shape =}")
    print(f"{env._max_episode_steps = }")

    # Print some basic information about the dataset
    print(f"Number of transitions: {len(dataset['rewards'])}")
    print(dataset.keys())
    print("-" * 40)
