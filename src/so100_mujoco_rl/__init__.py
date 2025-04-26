from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

register(
    id="Env01-v1",
    entry_point="so100_mujoco_rl.envs.env01_v1:Env01",
    max_episode_steps=4000,
    reward_threshold=6000,
)

register(
    id="Env02-v1",
    entry_point="so100_mujoco_rl.envs.env02_v1:Env02",
    max_episode_steps=6000,
    reward_threshold=8000,
)

register(
    id="Env03-v1",
    entry_point="so100_mujoco_rl.envs.env03_v1:Env03",
    max_episode_steps=6000,
    reward_threshold=8000,
)

register(
    id="Env04-v1",
    entry_point="so100_mujoco_rl.envs.env04_v1:Env04",
    max_episode_steps=6000,
    reward_threshold=8000,
)
