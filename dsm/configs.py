import dataclasses
import functools
from typing import Annotated

import fiddle as fdl
import flax.linen as nn
import jax.numpy as jnp
import optax
from fiddle import selectors
from fiddle.experimental import auto_config

from dsm import kernels, tags
from dsm.models import MLP, ResidualMLP
from dsm.state import SoftTargetParamsUpdate, TargetParamsUpdate
from dsm.types import DistanceFunction, Environment, Kernel, KernelFactory


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    seed: int

    env: Environment

    generator: Annotated[nn.Module, tags.Generator]
    generator_optim: Annotated[optax.GradientTransformation, tags.Generator]

    discriminator: Annotated[nn.Module, tags.Discriminator]
    discriminator_optim: Annotated[optax.GradientTransformation, tags.Discriminator]

    # data type for model params & computation
    dtype: jnp.dtype

    # state kernel
    inner_kernel: Kernel
    # Whether or not to use median heuristic for bandwidth
    inner_kernel_adaptive_bandwidth: bool
    inner_linear_kernel: bool
    # Use a single discriminator for each model atom?
    inner_separate_discriminator: bool
    inner_distance_fn: DistanceFunction

    # model kernel
    outer_kernel: Kernel
    # Whether or not to use median heuristic for bandwidth
    outer_kernel_adaptive_bandwidth: bool

    # Number of generator & discriminator steps per train step
    num_discriminator_steps: int
    num_generator_steps: int

    # Discount factor
    gamma: float
    # n-step horizon
    horizon: int
    # Whether to bootstrap or use trajectory samples
    bootstrap: bool

    # Dimension of input noise
    latent_dims: int
    # Number of model atoms
    num_outer: int
    # Number of state samples
    num_inner: int
    num_grad_updates: int
    batch_size: int
    target_params_update: TargetParamsUpdate

    log_every: int
    plot_every: int
    plot_num_samples: int

    # Include source state in SR, i.e., is cumulant 1 { S_t = s } or 1 { S_{t+1} = s }
    cumulant_is_source_state: bool

    # Whether or not to use outer MMD
    # i.e., distributional or not
    distributional: bool

    def __post_init__(self):
        if self.inner_kernel_adaptive_bandwidth:
            assert len(self.inner_kernel.bandwidths) == 1, (  # type: ignore
                "Adaptive bandwidth requires single bandwidth"
            )
        assert self.num_discriminator_steps >= 0
        assert self.num_generator_steps > 0
        assert 0.0 <= self.gamma < 1


@auto_config.auto_config
def base() -> Config:
    dtype = tags.DType.new(default=jnp.float32)

    return Config(
        seed=0,
        env="Pendulum-v1",
        generator=MLP(num_layers=3, num_hidden_units=256, dtype=dtype, param_dtype=dtype),
        generator_optim=optax.adam(
            learning_rate=auto_config.with_tags(6.25e-5, (tags.Generator, tags.LearningRate)),
            eps=1.5e-4,
        ),
        num_generator_steps=1,
        # iResMLP, i.e., spectral norm on a residual MLP for injectivity of the discriminator
        discriminator=nn.SpectralNorm(
            ResidualMLP(
                num_hidden_units=256,
                num_layers_per_block=2,
                num_blocks=2,
                num_outputs=8,
                dtype=dtype,
                param_dtype=dtype,
            ),
            collection_name="spectral_norm",
        ),
        discriminator_optim=optax.adam(
            learning_rate=auto_config.with_tags(6.25e-5, (tags.Discriminator, tags.LearningRate)),
            eps=1.5e-4,
        ),
        num_discriminator_steps=1,
        dtype=dtype,
        outer_kernel=kernels.InverseMultiQuadricKernel(bandwidths=(1,)),
        outer_kernel_adaptive_bandwidth=True,
        inner_kernel=kernels.RationalQuadraticKernel(bandwidths=(0.2, 0.5, 1.0, 2.0, 5.0)),
        inner_kernel_adaptive_bandwidth=False,
        inner_distance_fn=kernels.euclidean_distance,
        inner_separate_discriminator=False,
        inner_linear_kernel=False,
        gamma=0.95,
        horizon=6,
        bootstrap=True,
        latent_dims=8,
        num_outer=51,
        num_inner=32,
        num_grad_updates=2_500_000,
        batch_size=64,
        target_params_update=SoftTargetParamsUpdate(step_size=0.01),
        log_every=1_000,
        plot_every=100_000,
        plot_num_samples=1024,
        cumulant_is_source_state=True,
        distributional=True,
    )


def gamma_model_ensemble() -> fdl.Config[Config]:
    config = base.as_buildable()
    config.distributional = False
    config.inner_separate_discriminator = True
    return config


def single_step_model() -> fdl.Config[Config]:
    config = gamma_model_ensemble()
    config.cumulant_is_source_state = False
    config.num_outer = 1
    config.gamma = 0.0
    config.horizon = 1
    config.bootstrap = False

    return config


@auto_config.auto_config
def adversarial() -> Config:
    return base()


def median() -> fdl.Config[Config]:
    config = base.as_buildable()
    config.inner_kernel = fdl.Config(kernels.RBFKernel, bandwidths=(1,))
    config.outer_kernel = fdl.Config(kernels.RBFKernel, bandwidths=(1,))
    config.inner_kernel_adaptive_bandwidth = True
    config.num_discriminator_steps = 0
    config.discriminator_optim = fdl.Config(optax.identity)
    return config


# ==============================
# ========== Fiddlers ==========
# ==============================


def outer_kernel(config: fdl.Config[Config], kernel_factory: KernelFactory, bandwidths: tuple[float, ...]) -> None:
    selectors.select(config, tag=tags.OuterKernel).replace(kernel_factory(bandwidths=bandwidths))


def inner_kernel(config: fdl.Config[Config], kernel_factory: KernelFactory, bandwidths: tuple[float, ...]) -> None:
    selectors.select(config, tag=tags.InnerKernel).replace(kernel_factory(bandwidths=bandwidths))


outer_rbf_kernel = functools.partial(outer_kernel, kernel_factory=kernels.RBFKernel, bandwidths=(1,))
outer_cauchy_kernel = functools.partial(outer_kernel, kernel_factory=kernels.CauchyKernel, bandwidths=(1,))
outer_inverse_multi_quadric_kernel = functools.partial(
    outer_kernel, kernel_factory=kernels.InverseMultiQuadricKernel, bandwidths=(1,)
)

inner_rbf_kernel = functools.partial(inner_kernel, kernel_factory=kernels.RBFKernel, bandwidths=(1,))
inner_cauchy_kernel = functools.partial(inner_kernel, kernel_factory=kernels.CauchyKernel, bandwidths=(1,))
inner_inverse_multi_quadric_kernel = functools.partial(
    inner_kernel, kernel_factory=kernels.InverseMultiQuadricKernel, bandwidths=(1,)
)


def energy_distance_kernel(config: fdl.Config[Config]) -> None:
    config.inner_distance_fn = fdl.Partial(kernels.energy_distance)
    config.inner_kernel = fdl.Config(kernels.NegationKernel)
    config.num_discriminator_steps = 0


def inner_energy_distance_kernel(config: fdl.Config[Config], alpha: float = 1.5) -> None:
    config.inner_kernel = fdl.Config(kernels.NegationKernel)
    config.inner_distance_fn = fdl.Partial(kernels.energy_distance, alpha=alpha)


def mlp_discriminator(config: fdl.Config[Config]) -> None:
    config.discriminator = fdl.Config(
        MLP,
        num_layers=3,
        num_hidden_units=256,
        num_outputs=8,
        dtype=config.dtype,
        param_dtype=config.dtype,
    )
