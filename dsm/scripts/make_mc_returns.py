import concurrent.futures
import copy
import pathlib
import pickle
from typing import Annotated

import gymnasium as gym
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm.rich as tqdm
import tyro

from dsm import datasets, envs, plotting
from dsm.types import Environment


def single_mc_rollout(args):
    env, obs, policy_path, key = args

    policy = datasets.load_policy(policy_path)

    episode_steps = 0
    episode_obs: list[npt.NDArray] = []
    episode_actions: list[npt.NDArray] = []
    episode_return = 0.0
    action_t = None

    while True:
        key, action_t = policy(key, obs)

        episode_obs.append(obs)
        episode_actions.append(np.asarray(action_t))
        obs, reward, terminal, truncated, _ = env.step(action_t)
        episode_return += reward
        episode_steps += 1

        if terminal or truncated:
            print(f"Episode return: {episode_return:.2f}")
            return episode_return, (np.array(episode_obs), np.array(episode_actions))


def parallel_mc_rollout(
    original_env: gym.Env,
    original_obs: npt.NDArray,
    policy_path: pathlib.Path,
    rng: np.random.Generator,
    *,
    num_rollouts: int,
    with_multiprocessing: bool = False,
) -> tuple[npt.NDArray, list[tuple[npt.NDArray, npt.NDArray]]]:
    def generate_args():
        for _ in range(num_rollouts):
            key = jax.random.PRNGKey(rng.integers(0, 2**32))
            env = copy.deepcopy(original_env)
            obs = copy.deepcopy(original_obs)
            yield env, obs, policy_path, key

    # Run the rollouts in parallel
    if with_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            results = []
            mc_returns = []

            # Use executor.map() to run the rollouts in parallel
            for mc_return, result in tqdm.tqdm(executor.map(single_mc_rollout, generate_args()), total=num_rollouts):
                mc_returns.append(mc_return)
                results.append(result)
    else:
        results = []
        mc_returns = []

        for args in tqdm.tqdm(generate_args(), total=num_rollouts):
            mc_return, result = single_mc_rollout(args)
            mc_returns.append(mc_return)
            results.append(result)

    return np.array(mc_returns), results


def main(
    output: pathlib.Path,
    *,
    env_id: Annotated[Environment, tyro.conf.arg(name="env")],
    policy_path: Annotated[pathlib.Path, tyro.conf.arg(name="policy")],
    num_rollouts: int = 5_000,
    plot_mc_returns: bool = False,
):
    env = envs.make(env_id)
    rng = np.random.default_rng(0)

    for source_index, (source_state, source_obs) in enumerate(zip(*plotting.source_states(env_id))):
        env.reset()
        match env_id:
            case "Pendulum-v1":
                env.unwrapped.__setattr__("state", source_state)
                obs = source_obs  # pyright: ignore
            case str() if env_id.startswith("dm_control/"):
                env.unwrapped._env._physics = source_state  # type: ignore
                obs = source_obs
            case _:
                raise NotImplementedError
        mc_returns, observations_and_actions = parallel_mc_rollout(
            env,
            obs,
            policy_path,
            rng,
            num_rollouts=num_rollouts,
        )
        with (output / f"mc-returns-{source_index}.pkl").open("wb") as fp:
            pickle.dump(observations_and_actions, fp)

        if plot_mc_returns:
            fig, ax = plt.subplots()
            ax.hist(mc_returns)
            fig.savefig((output / f"mc-returns-default-{source_index}.png").as_posix())


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        tyro.cli(main)
