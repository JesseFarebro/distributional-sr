import functools
import pathlib
import pickle

import jax
import numpy.typing as npt
import tensorflow as tf
import tf2jax

from dsm.types import Environment, Policy, TransitionDataset

_POLICY_REGISTRY: dict[Environment, pathlib.Path] = {
    "Pendulum-v1": pathlib.Path("datasets/pendulum/sac/policy"),
}

_DATASET_REGISTRY: dict[Environment, pathlib.Path] = {
    "Pendulum-v1": pathlib.Path("datasets/pendulum/sac/dataset.pkl"),
}

_MC_DATASET_REGISTRY_WITH_TRUNCATION: dict[Environment, dict[float, pathlib.Path]] = {
    "Pendulum-v1": {
        0.95: pathlib.Path("datasets/pendulum/sac/mc-visitation-discount-0.95.pkl"),
    }
}

_MC_DATASET_REGISTRY_WITHOUT_TRUNCATION: dict[Environment, pathlib.Path] = {
    "Pendulum-v1": pathlib.Path("datasets/pendulum/sac/mc-visitation.pkl"),
}


def dataset_path_for_env(
    env: Environment,
    *,
    registry: dict[Environment, pathlib.Path],
) -> pathlib.Path:
    """Return the path to the dataset for the given environment."""
    if path := registry.get(env):
        return path
    raise ValueError(f"Unknown dataset for environment: {env}. Available datasets: {registry.keys()}")


@functools.cache
def make_dataset(env: Environment) -> TransitionDataset:
    """Make a dataset for the given environment."""
    with dataset_path_for_env(env, registry=_DATASET_REGISTRY).open("rb") as fp:
        return pickle.load(fp)


@functools.cache
def make_mc_dataset(
    env: Environment, *, discount: float | None = None
) -> list[tuple[npt.NDArray, list[tuple[npt.NDArray, npt.NDArray]]]]:
    """Make a MC dataset for the given environment."""
    if not discount:
        with _MC_DATASET_REGISTRY_WITHOUT_TRUNCATION[env].open("rb") as fp:
            return pickle.load(fp)

    with _MC_DATASET_REGISTRY_WITH_TRUNCATION[env][discount].open("rb") as fp:
        return pickle.load(fp)


@functools.cache
def load_policy(policy_path: pathlib.Path) -> Policy:
    policy = tf.saved_model.load(policy_path.as_posix())
    pure_policy_func, policy_params = tf2jax.convert_from_restored(getattr(policy, "__call__"))  # type: ignore

    @jax.jit
    def policy_fn(rng: jax.random.KeyArray, observation: jax.Array) -> tuple[jax.random.KeyArray, jax.Array]:
        rng, key = jax.random.split(rng)
        action, _ = pure_policy_func(policy_params, observation, key)
        return rng, action

    return policy_fn


def make_policy(env: Environment) -> Policy:
    """Make a policy for the given environment."""
    return load_policy(_POLICY_REGISTRY[env])
