from typing import Dict, Iterable, Mapping, Any, Callable
import argparse
import warnings

from stable_baselines3 import PPO  # type: ignore
import gym  # type: ignore

from . import env
from .util import log_range

Config = Mapping[str, Any]
ConfigSet = Iterable[Config]


default_config = argparse.Namespace(
    environment=env.NavToCenter,
    bottleneck_temperature=1.5,
    bottleneck_hard=False,
    pre_bottleneck_arch=[2**5, 2**6],
    post_bottleneck_arch=[2**5],
    policy_activation="tanh",
    eval_freq=5_000,
    total_timesteps=200_000,
    eval_episodes_logging=200,
    eval_episodes=10_000,
    rl_algorithm=PPO,
    n_steps=0x100,
    batch_size=0x40,
    learning_rate=3e-3,
    init_model_path=None,
    goal_radius=1.0,
    world_radius=9.0,
    eval_world_radius=9.0,
    # Maximum step count as a multiplier of world_radius
    max_step_scale=3.0,
    gamma=0.9,
    sparsity=1,
    biased_reward_shaping=False,
    n_dims=5,
    n_opts=1,
)


N_ITERS = 600


BASE_CFGS: Dict[str, Config] = {
    "nodyn": {
        "environment": env.NoDynamics,
    },
    "recon": {
        "environment": env.Reconstruction,
        "n_dims": 1,
    },
    "sig": {
        "environment": env.Signal,
        "n_dims": 5,
        "n_opts": 2,
    },
    "nav": {
        "environment": env.NavToCenter,
    },
}


def make_config_set(name: str, func: Callable) -> Mapping[str, ConfigSet]:
    return {f"{env}_{name}": func(base) for env, base in BASE_CFGS.items()}


def _timesteps(base: Config) -> ConfigSet:
    for x in log_range(1_000_000, 100, N_ITERS):
        ef = min(default_config.eval_freq, x // 5)
        yield {
            **base,
            "eval_freq": ef,
            "total_timesteps": int(x),
        }


def _learning_rate(base: Config) -> ConfigSet:
    for x in log_range(1e-4, 0.1, N_ITERS):
        yield {
            **base,
            "learning_rate": x,
        }


def _elcc_lexicon_size(base: Config) -> ConfigSet:
    for x in log_range(2**8, 2**3, 10):
        yield {
            **base,
            "pre_bottleneck_arch": [0x20, int(x)],
            "note": int(x),
        }


def _lexicon_size(base: Config) -> ConfigSet:
    for x in log_range(2**8, 2**3, N_ITERS):
        yield {
            **base,
            "pre_bottleneck_arch": [0x20, int(x)],
            "note": x,
        }


def _elcc_temperature(base: Config) -> ConfigSet:
    for x in log_range(0.1, 10, 10):
        yield {
            **base,
            "bottleneck_temperature": x,
        }


def _temperature(base: Config) -> ConfigSet:
    for x in log_range(0.1, 10, N_ITERS):
        yield {
            **base,
            "bottleneck_temperature": x,
        }


def _buffer_size(base: Config) -> ConfigSet:
    # Keep the number of gradient steps constant
    timesteps_base = base.get("total_timesteps", 100_000) // 0x100
    for x in log_range(2**1, 2**10, N_ITERS):
        # Avoid an issue with PPO
        if int(x) % default_config.batch_size == 1:
            x += 1
        timesteps = int(x) * timesteps_base
        eval_freq = min(default_config.eval_freq, timesteps // 5)
        yield {
            **base,
            "n_steps": int(x),
            "total_timesteps": timesteps,
            "eval_freq": eval_freq,
            # Prevent name collisions due to int(x)
            "note": x,
        }


CONFIGS: Mapping[str, ConfigSet] = {
    "quick_test": [
        {
            "learning_rate": x,
            "total_timesteps": 40_000,
        }
        for x in log_range(1e-4, 0.1, 4)
    ],
    **make_config_set("timesteps", _timesteps),
    **make_config_set("temperature", _temperature),
    **make_config_set("buffer_size", _buffer_size),
    **make_config_set("lexicon_size", _lexicon_size),
    **make_config_set("learning_rate", _learning_rate),
    **make_config_set("elcc_temperature", _elcc_temperature),
    **make_config_set("elcc_lexicon_size", _elcc_lexicon_size),
}
