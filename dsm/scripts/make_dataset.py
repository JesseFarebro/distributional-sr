import functools
import logging
import pathlib
import pickle
from typing import Annotated

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
import tqdm.rich as tqdm
import tyro
from jax.experimental import jax2tf
from sbx import PPO, SAC
from sbx.common.policies import BaseJaxPolicy

from dsm import datasets, envs, stade
from dsm.types import Environment


def make_sbx_model(env_id: Environment, env: gym.Env) -> PPO | SAC:
    match env_id:
        case "Pendulum-v1" | "dm_control/pendulum-swingup-v0":
            return SAC("MlpPolicy", env, verbose=1)
        case "dm_control/walker-walk-v0":
            return SAC("MlpPolicy", env, verbose=1)
        case "dm_control/hopper-stand-v0":
            return SAC("MlpPolicy", env, verbose=1)
        case _:
            raise ValueError(f"Unknown environment: {env.spec!r}")


def main(
    dataset_path: pathlib.Path,
    *,
    env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    seed: int | None = None,
    train_steps: int = 500_000,
    policy_path: pathlib.Path,
    num_eval_steps: int = 500_000,
    sticky_action_prob: float = 0.0,
    force: bool = False,
):
    env = envs.make(env_id)

    if force or not policy_path.exists():
        model = make_sbx_model(env_id, env)
        model.learn(total_timesteps=train_steps, progress_bar=True)

        @functools.partial(
            tf.function,
            input_signature=[
                tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype),  # pyright: ignore
                tf.TensorSpec((2,), np.uint32),  # pyright: ignore
            ],
        )
        @functools.partial(jax2tf.convert, with_gradient=False)
        def policy_apply(obs: np.ndarray, rng_key: np.ndarray) -> np.ndarray:
            return BaseJaxPolicy.sample_action(model.policy.actor_state, obs, rng_key)

        policy = tf.Module()
        policy.__call__ = policy_apply

        tf.saved_model.save(policy, policy_path.as_posix())

    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)

    # Convert saved model to Jax function
    rng = np.random.default_rng(seed)
    rng_key = jax.random.PRNGKey(rng.integers(0, 2**32))
    policy_func = datasets.load_policy(policy_path)

    action_t = None
    timestep_t = env.reset()
    transitions = [timestep_t]
    episode_index, episode_return = 0, 0.0
    for step in tqdm.tqdm(range(num_eval_steps)):
        rng_key, proposed_action = policy_func(rng_key, timestep_t.observation)
        if action_t is None or rng.uniform() >= sticky_action_prob:
            action_t = proposed_action
        timestep_t = env.step(action_t)  # pyright: ignore

        if not timestep_t.first():
            episode_return += timestep_t.reward  # pyright: ignore
        if timestep_t.last():
            logging.info(f"Episode {episode_index} return: {episode_return}")
            episode_index += 1
            episode_return = 0.0

        transitions.append(timestep_t)

    transitions = jax.tree_util.tree_map(lambda *arrs: np.vstack(arrs), *transitions)
    with dataset_path.open("wb") as fp:
        pickle.dump(transitions, fp)


if __name__ == "__main__":
    tyro.cli(main)
