import functools
import pathlib
from typing import Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dsm import plotting, rewards
from dsm.configs import Config
from dsm.plotting.utils import sample_from_sr
from dsm.state import FittedValueTrainState
from dsm.train import load_state_and_config


@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "num_steps",
    ),
)
def rollout(
    rng: jax.Array, *, model: FittedValueTrainState, source_state: jax.Array, config: Config, num_steps: int
) -> jax.Array:
    def one_step(x: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
        key = jax.random.fold_in(rng, i)
        x = jnp.squeeze(sample_from_sr(model, key, x, num_samples=1, num_outer=1, num_latent_dims=config.latent_dims))
        return x, x

    return jax.lax.scan(one_step, source_state, jnp.arange(0, num_steps, 1))[1]


@functools.partial(jax.jit, static_argnames=("config", "num_steps", "num_atoms", "reward_fn"))
def mc_return_distribution(
    rng: jax.Array,
    model: FittedValueTrainState,
    source_state: jax.Array,
    config: Config,
    num_steps: int,
    # /,
    num_atoms: int,
    reward_fn: Callable,
) -> jax.Array:
    _rollout = functools.partial(rollout, model=model, source_state=source_state, config=config, num_steps=num_steps)
    traces = jax.vmap(_rollout)(jax.random.split(rng, num_atoms))
    rewards = jax.vmap(jax.vmap(reward_fn))(traces, jnp.zeros((traces.shape[0], traces.shape[1], 1)))
    discounts = config.gamma ** jnp.arange(num_steps)
    discounts = einops.repeat(discounts, "num_steps -> num_trajectories num_steps", num_trajectories=num_atoms)
    return einops.einsum(rewards.squeeze(), discounts.squeeze(), "n_atoms n_steps, n_atoms n_steps -> n_atoms")


def main(
    output: pathlib.Path,
    *,
    checkpoint_path: pathlib.Path,
    num_atoms: int,
    seed: int = 42,
    num_steps: int = 200,
):
    rng = jax.random.PRNGKey(seed)
    state, config = load_state_and_config(checkpoint_path)
    env_id = "Pendulum-v1"
    for reward_index, (reward_name, reward_fn) in enumerate(getattr(rewards, env_id).items()):
        for source_index, (source_state, source_obs) in enumerate(zip(*plotting.source_states(env_id))):
            rng, ret_dist_rng = jax.random.split(rng)
            return_distribution = mc_return_distribution(
                ret_dist_rng, state, source_obs, config, num_steps, reward_fn=reward_fn, num_atoms=num_atoms
            )

            with (output / f"returns-{source_index}-{reward_index}.npz").open("wb") as fp:
                np.savez(fp, dsr=np.asarray(return_distribution))


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        tyro.cli(main)
