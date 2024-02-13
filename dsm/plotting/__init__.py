import functools
import logging
import pathlib
import pickle
import typing
from typing import Any, Protocol

import jax
import numpy.typing as npt

from dsm import rewards
from dsm.configs import Config
from dsm.plotting import pendulum, utils
from dsm.state import FittedValueTrainState
from dsm.types import Environment


@typing.runtime_checkable
class PlotProtocol(Protocol):
    def source_states(self) -> tuple[list[Any], list[npt.NDArray]]: ...

    def plot_samples(
        self,
        source: npt.NDArray,
        state: FittedValueTrainState,
        rng: jax.random.KeyArray,
        *,
        config: Config,
    ) -> npt.NDArray | None: ...


_PLOT_BY_ENVIRONMENT: dict[Environment, PlotProtocol] = {
    "Pendulum-v1": pendulum,
}


@functools.cache
def source_states(
    environment: Environment,
) -> tuple[list[Any], list[npt.NDArray]]:
    if environment in _PLOT_BY_ENVIRONMENT:
        return _PLOT_BY_ENVIRONMENT[environment].source_states()

    if not environment.startswith("dm_control/"):
        raise ValueError(f"Unknown environment: {environment!r}")
    env_name = environment.removeprefix("dm_control/").removesuffix("-v0")

    initial_states = pathlib.Path(f"datasets/{env_name}/initial-states")

    states = []
    observations = []
    for path in initial_states.glob("*.pkl"):
        with path.open("rb") as f:
            physics, obs = pickle.load(f)
            states.append(physics)
            observations.append(obs)

    return states, observations


def plot_samples(
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    *,
    config: Config,
) -> dict[str, npt.NDArray]:
    if config.env in _PLOT_BY_ENVIRONMENT:
        plot_module = _PLOT_BY_ENVIRONMENT[config.env]
        plots = {
            f"mmd-samples/{source.tolist()}": plot_module.plot_samples(source, state, rng, config=config)
            for (_, source) in zip(*plot_module.source_states())
        }
        return {k: v for k, v in plots.items() if v is not None}

    return {}


def plot_cdf(
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    *,
    config: Config,
) -> dict[str, npt.NDArray]:
    plot_module = _PLOT_BY_ENVIRONMENT[config.env]
    images: dict[str, npt.NDArray] = {}

    for reward_fn_name, reward_fn in getattr(rewards, config.env).items():
        logging.info(f"Plotting CDF for {config.env} reward function {reward_fn_name}")
        for idx, (_, source) in enumerate(zip(*plot_module.source_states())):
            images[f"cdf/{reward_fn_name}/{idx}"] = utils.plot_cdf(source, reward_fn, state, rng, config=config)
    return images
