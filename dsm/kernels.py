import dataclasses
import functools
import typing
from typing import Literal

import jax
import jax.numpy as jnp

from dsm.types import DistanceFunction, Kernel


def pairwise_distance(x: jax.Array, y: jax.Array, *, distance_fn: DistanceFunction) -> jax.Array:
    return jax.vmap(lambda x1: jax.vmap(lambda y1: distance_fn(x1, y1))(y))(x)


@typing.overload
def mmd(
    x: jax.Array,
    y: jax.Array,
    *,
    kernel: Kernel,
    distance_fn: DistanceFunction,
    from_samples: bool = ...,
    adaptive_bandwidth: bool = ...,
    return_distances: Literal[True] = ...,
    with_linear_kernel: bool = ...,
) -> tuple[jax.Array, jax.Array]: ...


@typing.overload
def mmd(
    x: jax.Array,
    y: jax.Array,
    *,
    kernel: Kernel,
    distance_fn: DistanceFunction,
    from_samples: bool = ...,
    adaptive_bandwidth: bool = ...,
    return_distances: Literal[False] = ...,
    with_linear_kernel: bool = ...,
) -> jax.Array: ...


@functools.partial(
    jax.jit,
    static_argnames=(
        "kernel",
        "distance_fn",
        "from_samples",
        "return_distances",
        "adaptive_bandwidth",
        "with_linear_kernel",
    ),
)
def mmd(
    x: jax.Array,
    y: jax.Array,
    *,
    kernel: Kernel,
    distance_fn: DistanceFunction,
    from_samples: bool = True,
    adaptive_bandwidth: bool = False,
    return_distances: bool = False,
    with_linear_kernel: bool = False,
) -> tuple[jax.Array, jax.Array] | jax.Array:
    nx = x.shape[0]
    ny = y.shape[0]

    dxx = pairwise_distance(x, x, distance_fn=distance_fn)
    dyy = pairwise_distance(y, y, distance_fn=distance_fn)
    dxy = pairwise_distance(x, y, distance_fn=distance_fn)

    bandwidth = 1.0
    if adaptive_bandwidth:
        all_distances = jnp.concatenate([
            dxx.at[jnp.diag_indices_from(dxx)].set(jnp.nan),
            dyy.at[jnp.diag_indices_from(dyy)].set(jnp.nan),
            dxy,
        ])

        bandwidth = jnp.maximum(jnp.nanmedian(all_distances), 1e-6)
        bandwidth = jax.lax.stop_gradient(bandwidth)

    kxx = kernel(dxx / bandwidth)
    kyy = kernel(dyy / bandwidth)
    kxy = kernel(dxy / bandwidth)

    if with_linear_kernel:
        kxx += pairwise_distance(x, x, distance_fn=jnp.dot)  # type: ignore
        kyy += pairwise_distance(y, y, distance_fn=jnp.dot)  # type: ignore
        kxy += pairwise_distance(x, y, distance_fn=jnp.dot)  # type: ignore

    mxy = kxy.sum() / (nx * ny)
    if from_samples:
        mxx = (kxx - jnp.diag(jnp.diag(kxx))).sum() / (nx * (nx - 1))
        myy = (kyy - jnp.diag(jnp.diag(kyy))).sum() / (ny * (ny - 1))
    else:
        mxx = kxx.sum() / (nx * nx)
        myy = kyy.sum() / (ny * ny)

    distance = mxx + myy - 2 * mxy  # pyright: ignore
    if return_distances:
        return distance, jnp.concatenate([dxx, dyy, dxy])
    return distance


def euclidean_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.sum((x - y) ** 2)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def norm_alpha(x: jax.Array, alpha: float = 1.0) -> jax.Array:
    """
    Implements the function x -> ||x||^alpha
    Same thing as (||x||^2)^(alpha / 2)
    Gradient is (alpha / 2) ||x||^2(alpha / 2 - 1) * 2x = alpha * ||x||^(alpha - 2)
    """
    return jnp.sum(jnp.square(x)) ** (alpha / 2)


@norm_alpha.defjvp
def _norm_alpha_dx(alpha: float, primal: tuple[jax.Array], tangent: tuple[jax.Array]) -> tuple[jax.Array, jax.Array]:
    (x,) = primal
    (xdot,) = tangent
    grad_problematic = jnp.all(jnp.logical_not(x))
    x_processed = jnp.where(grad_problematic, jnp.ones_like(x), x)
    grad = alpha * (jnp.linalg.norm(x_processed) ** (alpha - 2)) * x_processed.dot(xdot)
    ans = norm_alpha(x, alpha=alpha)
    return ans, jnp.where(grad_problematic, 0.0, grad)  # type: ignore


def energy_distance(x: jax.Array, y: jax.Array, alpha: float = 1.0) -> jax.Array:
    return norm_alpha(x - y, alpha=alpha)


def mmd_distance(x: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
    # mmd is non-negative, but we could have numerical errors when mmd ~ 0, so just clip it.
    return jnp.maximum(mmd(x, y, **kwargs), 0.0)


@dataclasses.dataclass(frozen=True)
class NegationKernel:
    def __call__(self, zz: jax.Array) -> jax.Array:
        return -zz


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _custom_fixed_raise_alpha(x: jax.Array, alpha: float = 1.0) -> jax.Array:
    return x**alpha


@_custom_fixed_raise_alpha.defjvp
def _custom_fixed_raise_alpha_dx(
    alpha: float, primal: tuple[float], tangents: tuple[float]
) -> tuple[jax.Array, jax.Array]:
    (x,) = primal
    (xdot,) = tangents
    gradient_problematic = jnp.all(jnp.logical_not(x))
    x_processed = jnp.where(gradient_problematic, 1.0, x)
    ans = _custom_fixed_raise_alpha(x, alpha)
    grad = alpha * (x_processed ** (alpha - 1)) * xdot
    return ans, jnp.where(gradient_problematic, 0.0, grad)


@dataclasses.dataclass(frozen=True)
class EnergyDistanceKernel:
    alpha: float

    def __call__(self, zz: jax.Array) -> jax.Array:
        return -_custom_fixed_raise_alpha(zz, self.alpha)


@dataclasses.dataclass(frozen=True)
class RBFKernel:
    bandwidths: tuple[float, ...]

    def __call__(self, zz: jax.Array) -> jax.Array:
        bandwidths = jnp.array(self.bandwidths)
        return jnp.sum(jax.vmap(lambda b: jnp.exp(-zz / b))(bandwidths), axis=0)


@dataclasses.dataclass(frozen=True)
class CauchyKernel:
    bandwidths: tuple[float, ...]

    def __call__(self, zz: jax.Array) -> jax.Array:
        bandwidths = jnp.array(self.bandwidths)
        return jnp.sum(jax.vmap(lambda b: jax.lax.reciprocal(1 + zz / b))(bandwidths), axis=0)


@dataclasses.dataclass(frozen=True)
class InverseMultiQuadricKernel:
    bandwidths: tuple[float, ...]

    def __call__(self, zz: jax.Array) -> jax.Array:
        bandwidths = jnp.array(self.bandwidths)
        return jnp.sum(jax.vmap(lambda b: jax.lax.rsqrt(1 + zz / b))(bandwidths), axis=0)


@dataclasses.dataclass(frozen=True)
class RationalQuadraticKernel:
    bandwidths: tuple[float, ...]

    def __call__(self, zz: jax.Array) -> jax.Array:
        bandwidths = jnp.array(self.bandwidths)
        return jnp.sum(jax.vmap(lambda b: (1 + zz / (2 * b)) ** (-b))(bandwidths), axis=0)
