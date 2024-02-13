from typing import Callable

import jax
import jax.numpy as jnp

from dsm.types import Environment


def _angle_normalize(x: jax.Array) -> jax.Array:
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def pendulum_default(s: jax.Array, a: jax.Array) -> jax.Array:
    max_torque = 2.0
    a = jnp.clip(a, -max_torque, max_torque)[0]
    return pendulum_no_action_penalty(s, a) - 0.001 * (a**2)


def pendulum_no_action_penalty(s: jax.Array, a: jax.Array) -> jax.Array:
    del a
    if len(s) == 3:
        t = jnp.arctan2(s[1], s[0])
    else:
        t = s[0]
    return -(_angle_normalize(t) ** 2 + 0.1 * s[-1] ** 2)


def pendulum_above_horizon(s: jax.Array, a: jax.Array) -> jax.Array:
    if len(s) == 3:
        t = jnp.arctan2(s[1], s[0])
    else:
        t = s[0]
    theta = _angle_normalize(t)
    below_horizon = jnp.abs(theta) > jnp.pi / 2
    return -(below_horizon + 0.1 * a**2)


def pendulum_counterclockwise(s: jax.Array, a: jax.Array) -> jax.Array:
    del a
    return jnp.minimum(s[1], 0)


def pendulum_clockwise(s: jax.Array, a: jax.Array) -> jax.Array:
    del a
    return jnp.maximum(s[1], 0)


def pendulum_angular_velocity_penalty(s: jax.Array, a: jax.Array) -> jax.Array:
    del a
    return -jnp.abs(s[-1])


def pendulum_counterclockwise_av(s: jax.Array, a: jax.Array) -> jax.Array:
    del a
    return jax.lax.select(s[-1] > 0, 1, 0)


def windy_gridworld_quadrants(s: jax.Array, a: jax.Array) -> jax.Array:
    del a
    x, y = s[0], s[1]

    return jax.lax.select(
        jnp.logical_and(x > 0, y > 0),
        1.0,
        jax.lax.select(
            jnp.logical_and(x > 0, y < 0),
            -2.0,
            jax.lax.select(
                jnp.logical_and(x < 0, y > 0),
                2.0,
                jax.lax.select(
                    jnp.logical_and(x < 0, y < 0),
                    -1.0,
                    0.0,
                ),
            ),
        ),
    )


_CUSTOM_REWARDS: dict[str, dict[str, Callable[..., jax.Array] | Callable[[], Callable[..., jax.Array]]]] = {
    "Pendulum-v1": {
        "Default": pendulum_no_action_penalty,
        "No Action Penalty": pendulum_no_action_penalty,
        "Above Horizon": pendulum_above_horizon,
        "Stay Left": pendulum_counterclockwise,
        "Stay Right": pendulum_clockwise,
        "Angular Velocity Penalty": pendulum_angular_velocity_penalty,
        "Counter Clockwise Penalty": pendulum_counterclockwise_av,
    },
    "WindyGridWorld-v0": {
        "Quadrants": windy_gridworld_quadrants,
    },
    "WindyGridWorld-top-v0": {
        "Quadrants": windy_gridworld_quadrants,
    },
    "WindyGridWorld-bottom-v0": {
        "Quadrants": windy_gridworld_quadrants,
    },
}


def __getattr__(env: Environment) -> dict[str, Callable[..., jax.Array]]:
    return _CUSTOM_REWARDS[env]  # type: ignore


def __dir__() -> list[str]:
    return list(_CUSTOM_REWARDS.keys())
