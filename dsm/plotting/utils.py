import functools
import logging
from typing import Callable

import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import numpy.typing as npt

from dsm import datasets
from dsm.configs import Config
from dsm.state import FittedValueTrainState
from dsm.types import Environment, Policy


@functools.partial(jax.jit, static_argnames=("num_outer", "num_samples", "num_latent_dims"))
def sample_from_sr(
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    source_state: jax.Array,
    *,
    num_samples: int,
    num_outer: int,
    num_latent_dims: int,
) -> jax.Array:
    zs = jax.random.normal(rng, (num_samples, num_outer, num_latent_dims))
    context = einops.repeat(source_state, "s -> i o s", i=num_samples, o=num_outer)
    xs = jnp.concatenate((zs, context), axis=-1)
    ys = jax.vmap(state.apply_fn, in_axes=(None, 0))(state.params, xs)
    return einops.rearrange(ys, "i o s -> o i s")


@functools.partial(jax.jit, static_argnames=("config", "num_samples", "policy", "reward_fn"))
def return_distribution(
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    source_state: jax.Array,
    *,
    policy: Policy,
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
    num_samples: int,
    config: Config,
) -> jax.Array:
    sample_rng, action_rng = jax.random.split(rng)
    samples = sample_from_sr(
        state,
        sample_rng,
        source_state,
        num_samples=num_samples,
        num_outer=config.num_outer,
        num_latent_dims=config.latent_dims,
    )
    keys = jax.random.split(action_rng, np.prod(samples.shape[:-1]))
    keys = jnp.array(keys).reshape(*samples.shape[:-1], -1)
    actions = jax.vmap(jax.vmap(policy))(keys, samples)
    rewards = jax.vmap(jax.vmap(reward_fn))(samples, actions[1]).squeeze()
    assert isinstance(config.gamma, float)
    return rewards.mean(axis=-1) / (1.0 - config.gamma)


def empirical_cdf(data: np.ndarray, bins: int):
    count, bins_count = np.histogram(data, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf


@functools.partial(jax.jit, static_argnames=("reward_fn", "discount"))
def _compute_mc_returns(
    observations_and_actions: tuple[jax.Array, jax.Array],
    *,
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
    discount: float | None = None,
) -> jax.Array:
    if discount is None:
        discount = 1.0
    rewards = jax.vmap(reward_fn)(*observations_and_actions)
    discounts = jnp.array([discount]) ** jnp.arange(rewards.shape[0])
    return jnp.sum(jnp.squeeze(rewards) * jnp.squeeze(discounts))


def compute_mc_returns(
    env: Environment,
    source: npt.NDArray,
    *,
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
    with_truncation: bool = False,
    config: Config,
) -> npt.NDArray:
    mc_dataset = datasets.make_mc_dataset(env, discount=config.gamma if with_truncation else None)
    mc_episodes = more_itertools.one(filter(lambda x: np.isclose(x[0], source).all(), mc_dataset))[1]

    logging.info("Comptuing MC returns for %d episodes", len(mc_episodes))
    return np.array([
        _compute_mc_returns(episode, reward_fn=reward_fn, discount=None if with_truncation else config.gamma)
        for episode in mc_episodes
    ])


def plot_cdf(
    source: npt.NDArray,
    reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    *,
    config: Config,
) -> npt.NDArray:
    policy = datasets.make_policy(config.env)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6 * 2, 6))

    dsr_returns = return_distribution(
        state,
        rng,
        source,
        policy=policy,
        reward_fn=reward_fn,
        num_samples=config.plot_num_samples,
        config=config,
    )
    dsr_cdf_xs, dsr_cdf_ys = empirical_cdf(dsr_returns, bins=config.num_outer)

    # Compute MC returns for non-truncated visitation distribution
    # Load MC returns with no discount factor applied (i.e., no truncation)
    mc_returns = compute_mc_returns(
        config.env,
        source,
        reward_fn=reward_fn,
        with_truncation=False,
        config=config,
    )
    mc_cdf_xs, mc_cdf_ys = empirical_cdf(mc_returns, bins=config.num_outer)

    # CDF Plot
    axs[0].plot(mc_cdf_xs, mc_cdf_ys, label="Monte-Carlo Returns")
    axs[0].plot(dsr_cdf_xs, dsr_cdf_ys, label="DSR Returns")
    axs[0].set_title("MC Return Distribution CDF")
    axs[0].legend()
    axs[0].set_xlabel("Z")
    axs[0].set_ylabel("P(Return <= Z)")

    # Histogram Plot
    axs[1].hist(mc_returns, bins=100, label="Monte-Carlo Returns", alpha=0.5, density=True)
    axs[1].hist(dsr_returns, bins=config.num_outer, label="DSR Returns", alpha=0.5, density=True)
    axs[1].set_title("MC Return Distribution Histogram")
    axs[1].set_xlabel("Z")
    axs[1].set_ylabel("Density")

    image = fig_to_ndarray(fig)
    plt.close(fig)

    return image


def fig_to_ndarray(fig: plt.Figure) -> npt.NDArray:  # type: ignore
    canvas = fig.canvas
    width, height = canvas.get_width_height()
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")  # type: ignore
    return image_array.reshape(height, width, 3)
