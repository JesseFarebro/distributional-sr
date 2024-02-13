import enum
import functools
import logging
import math
import pathlib
import typing
from typing import Callable, Type, TypeVar

import chex
import dm_env
import einops
import fiddle as fdl
import jax
import jax.numpy as jnp
import orbax.checkpoint
from clu import metrics as clu_metrics
from dm_env import specs
from fiddle.experimental import serialization
from flax import linen as nn
from jax.experimental import host_callback as hcb

from dsm import kernels, stade
from dsm.configs import Config
from dsm.state import FittedValueTrainState, State, TrainState
from dsm.types import PyTree, TransitionDataset

T = TypeVar("T")


def identity(x: T) -> T:
    return x


class DistributionalSRGenerator(nn.Module):
    model: nn.Module
    num_atoms: int
    num_state_dims: int

    @nn.jit
    @nn.compact
    def __call__(self, zs: jax.Array) -> jax.Array:
        return nn.vmap(
            lambda model, z: model(z, num_outputs=self.num_state_dims),
            in_axes=0,  # pyright: ignore
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_atoms,
        )(self.model, zs)


class DistributionalSRDiscriminator(nn.Module):
    model: nn.Module
    num_atoms: int
    with_separate_discriminator: bool = False

    @nn.compact
    def __call__(self, xs: jax.Array) -> jax.Array:
        vmap = functools.partial(
            nn.vmap,
            variable_axes={"params": None, "spectral_norm": None},
            split_rngs={"params": False, "spectral_norm": False},
        )

        model_vmap_kwargs = {}
        batch_vmap_kwargs = {}
        if self.with_separate_discriminator:
            model_vmap_kwargs |= dict(
                variable_axes={"params": 0, "spectral_norm": 0},
                split_rngs={"params": True, "spectral_norm": True},
                metadata_params={nn.meta.PARTITION_NAME: "outer"},
                axis_size=self.num_atoms,
            )
            batch_vmap_kwargs |= dict(
                variable_axes={"params": 0, "spectral_norm": 0},
                split_rngs={"params": False, "spectral_norm": False},
            )

        return vmap(
            vmap(
                vmap(lambda model, z: model(z)),
                **model_vmap_kwargs,
            ),
            **batch_vmap_kwargs,
        )(self.model, xs)


class TrainStepType(enum.IntEnum):
    DISCRIMINATOR = 0
    GENERATOR = 1


@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "update_which",
    ),
    donate_argnums=(0,),
)
@chex.assert_max_traces(2)
def train_step(
    state: State,
    batch: TransitionDataset,
    rng: jax.random.KeyArray,
    *,
    update_which: TrainStepType,
    config: Config,
) -> State:
    def _cast_batch_to_dtype(path: jax.tree_util.GetAttrKey, x: jax.Array) -> jax.Array:
        if path == (jax.tree_util.GetAttrKey("observation"),):
            return x.astype(config.dtype)
        return x

    batch = jax.tree_util.tree_map_with_path(_cast_batch_to_dtype, batch)

    def sample(g_params: PyTree[jax.Array], context: jax.Array, key: jax.random.KeyArray) -> jax.Array:
        """context: conditioning context"""
        context = einops.repeat(context, "b s -> b i o s", i=config.num_inner, o=config.num_outer)
        zs = jax.random.normal(key, (config.batch_size, config.num_inner, config.num_outer, config.latent_dims))
        xs = jnp.concatenate((zs, context), axis=-1)
        ys = jax.vmap(jax.vmap(functools.partial(state.generator.apply_fn, g_params)))(xs)
        # (b)atch, (i)nner, (o)utter
        return einops.rearrange(ys, "b i o s -> b o i s")

    def grad_fn(
        g_params: PyTree[jax.Array],
        d_params: PyTree[jax.Array],
        *,
        sign: float,
    ) -> tuple[jax.Array, tuple[dict[str, jax.Array], PyTree[jax.Array] | None]]:
        infos = {}
        horizon_rng, lhs_rng, rhs_rng = jax.random.split(rng, 3)
        # Sample from the state-discounted occupancy given the start state
        lhs = sample(g_params, batch.observation[:, 0, :], lhs_rng)
        infos["observation"] = jnp.mean(lhs)

        # Sample a horizon for each trajectory
        # This is unbounded but we'll truncate it later.
        horizons = jax.random.geometric(
            horizon_rng,
            1.0 - config.gamma,
            shape=(config.batch_size, config.num_outer, config.num_inner),
        )

        # We'll need to check if there's a terminal state along this trajectory
        # If there is a terminal state we'll have to truncate the horizon
        # There's two cases:
        #   1) If there's a terminal state and the sampled horizon is less than the
        #      index of the terminal state then we'll use the sampled horizon.
        #   2) If there's a terminal state and the sampled horizon is greater than the
        #      index of the terminal state then we'll use the index of the terminal state.
        # Otherwise we'll use the sampled horizon.
        terminals = einops.rearrange(batch.step_type == dm_env.StepType.LAST, "b h () -> b () () h")
        horizons = jnp.where(
            jnp.max(terminals, axis=-1),
            jnp.minimum(jnp.argmax(terminals, axis=-1) + (1 if config.cumulant_is_source_state else 0), horizons),
            horizons,
        )

        # We'll now construct the target sequence.
        # We'll append our bootstrapped target to the end of the trajectory,
        # e.g., the sampled trajectory from our dataset will be: [s_0, s_1, ..., s_T]
        # and we'll construct the target sequence as [s_0, s_1, ..., Psi(s_T)]
        if not config.bootstrap:
            rhs_observation_sequence = einops.repeat(
                batch.observation, "b h s -> b o i h s", o=config.num_outer, i=config.num_inner
            )
        else:
            # Sample from the state-discounted occupancy using our target parameters
            # from the last state in the sequence. This may or may not be used in the
            # update depending on the sampled horizon and whether or not there's a terminal
            # state in this trajectory.
            rhs = sample(state.generator.target_params, batch.observation[:, -1, :], rhs_rng)
            rhs_observation_sequence = jnp.concatenate(
                (
                    einops.repeat(
                        batch.observation[:, :-1, :], "b h s -> b o i h s", o=config.num_outer, i=config.num_inner
                    ),
                    # It's not as simple as just appending the bootstrapped target at the end of the sequence
                    # We must take into account whether or not a terminal state occured.
                    # There's two cases:
                    #   1) If there's a terminal state anywhere in the trajectory then we don't bootstrap.
                    #      This is because either:
                    #       i) horizon >= |trajectory| and since we're terminal we never want to bootstrap.
                    #       ii) horizon < |trajectory| hence we'll never use the bootstrapped target.
                    #   2) If there's no terminal state then we'll always want to bootstrap.
                    #
                    jnp.where(
                        einops.rearrange(jnp.max(terminals, axis=-1, keepdims=True), "b o i h -> b o i h ()"),
                        einops.repeat(
                            batch.observation[:, -1, :], "b s -> b o i () s", o=config.num_outer, i=config.num_inner
                        ),
                        einops.rearrange(rhs, "b o i s -> b o i () s"),
                    ),
                ),
                axis=-2,
            )
        # We'll now select the target sample from the sequence.
        # mode="clip" will allow us to select the bootstrapped sample
        # if the horizon is greater than the length of the trajectory.
        horizons = einops.rearrange(horizons, "b o i -> b o i () ()")
        if config.cumulant_is_source_state:
            horizons -= 1

        rhs = jnp.take_along_axis(
            rhs_observation_sequence,
            # einops.rearrange(horizons, "b o i -> b o i () ()") - 1,
            horizons,
            axis=-2,
            mode="clip",
        )
        # Squeeze out the horizon dimension.
        rhs = einops.rearrange(rhs, "b o i () s -> b o i s")

        d_updates = {}
        if config.num_discriminator_steps > 0:
            # # If we're updating the discriminator that'll mean we're applying
            # # an embedding to the generator's output.
            lhs, d_updates = state.discriminator.apply_fn(
                {"params": d_params, **state.discriminator.variables},  # type: ignore
                lhs,
                mutable=["spectral_norm"],
            )
            rhs, d_updates = state.discriminator.apply_fn(
                {"params": d_params, **d_updates},  # type: ignore
                rhs,
                mutable=["spectral_norm"],
            )
            infos["embedding"] = jnp.mean(jnp.linalg.norm(lhs, axis=-1))

        # Compute MMD loss
        if config.distributional:
            # Distributional loss with outer MMD
            loss = jax.vmap(
                functools.partial(
                    kernels.mmd,
                    kernel=config.outer_kernel,
                    distance_fn=functools.partial(
                        kernels.mmd_distance,
                        kernel=config.inner_kernel,
                        distance_fn=config.inner_distance_fn,
                        from_samples=True,
                        adaptive_bandwidth=config.inner_kernel_adaptive_bandwidth,
                        with_linear_kernel=config.inner_linear_kernel,
                    ),
                    from_samples=False,
                    adaptive_bandwidth=config.outer_kernel_adaptive_bandwidth,
                    return_distances=False,
                )
            )(lhs, rhs)
        else:
            # Non-distributional loss, i.e., ensemble of gamma models
            loss = jax.vmap(
                jax.vmap(
                    functools.partial(
                        kernels.mmd,
                        kernel=config.inner_kernel,
                        distance_fn=config.inner_distance_fn,
                        from_samples=True,
                        adaptive_bandwidth=config.inner_kernel_adaptive_bandwidth,
                        with_linear_kernel=config.inner_linear_kernel,
                        return_distances=False,
                    )
                )
            )(lhs, rhs)

        # Flip the sign of the loss depending on the type of update
        loss = jnp.mean(loss)
        infos["mmd"] = loss
        loss *= sign

        return loss, (infos, d_updates)

    d_grad_fn = jax.grad(functools.partial(grad_fn, sign=-1), argnums=1, has_aux=True)
    g_grad_fn = jax.grad(functools.partial(grad_fn, sign=1), argnums=0, has_aux=True)

    def d_update_step(d_state_t: TrainState) -> TrainState:
        d_grads, (d_infos, d_updates) = d_grad_fn(state.generator.params, d_state_t.params)  # type: ignore
        d_metrics = d_state_t.metrics.merge(d_state_t.metrics.single_from_model_output(**d_infos))
        return d_state_t.apply_gradients(grads=d_grads, metrics=d_metrics, variables=d_updates)

    def g_update_step(g_state_t: FittedValueTrainState) -> FittedValueTrainState:
        g_grads, (g_infos, _) = g_grad_fn(g_state_t.params, state.discriminator.params)
        g_metrics = g_state_t.metrics.merge(g_state_t.metrics.single_from_model_output(**g_infos))
        return g_state_t.apply_gradients(grads=g_grads, metrics=g_metrics)

    g_state = state.generator
    d_state = state.discriminator
    match update_which:
        case TrainStepType.DISCRIMINATOR:
            d_state = d_update_step(d_state)
        case TrainStepType.GENERATOR:
            g_state = g_update_step(g_state)

    return state.replace(generator=g_state, discriminator=d_state)


@functools.partial(
    jax.jit,
    static_argnames=(
        "metrics_callback",
        "config",
    ),
)
@chex.assert_max_traces(1)
def train_loop(
    state: State,
    data: TransitionDataset,
    rng: jax.random.KeyArray,
    *,
    num_steps: int,
    metrics_callback: Callable[[dict[str, jax.Array]], None] = lambda _: None,
    config: Config,
) -> State:
    def _reset_metrics(state_t: State) -> State:
        hcb.call(
            metrics_callback,
            {"step": state_t.step, **state_t.generator.metrics.merge(state_t.discriminator.metrics).compute()},
        )
        return state_t.replace(
            generator=state_t.generator.replace(metrics=state_t.generator.metrics.empty()),
            discriminator=state_t.discriminator.replace(metrics=state_t.discriminator.metrics.empty()),
        )

    def _sample_batch(key_t: jax.random.KeyArray) -> TransitionDataset:
        indices = jax.random.randint(key_t, (config.batch_size,), 0, len(data.observation) - config.horizon - 1)
        return jax.tree_util.tree_map(
            lambda op: jax.vmap(lambda index: jax.lax.dynamic_slice_in_dim(op, index, 1 + config.horizon, axis=0))(
                indices
            ),
            data,
        )

    def _train_step(state_t: State, key_t: jax.random.KeyArray, *, which: TrainStepType) -> tuple[State, None]:
        batch_key_t, train_key_t = jax.random.split(key_t)
        batch_t = _sample_batch(batch_key_t)
        return train_step(state_t, batch_t, train_key_t, update_which=which, config=config), None

    d_train_step = functools.partial(_train_step, which=TrainStepType.DISCRIMINATOR)
    g_train_step = functools.partial(_train_step, which=TrainStepType.GENERATOR)

    def _loop_step(index: int, state_t: State) -> State:
        key_t = jax.random.fold_in(rng, index)

        if config.num_discriminator_steps > 0:
            key_t, *d_keys = jax.random.split(key_t, config.num_discriminator_steps + 1)
            state_t, _ = jax.lax.scan(d_train_step, state_t, jnp.asarray(d_keys))

        key_t, *g_keys = jax.random.split(key_t, config.num_generator_steps + 1)
        state_t, _ = jax.lax.scan(g_train_step, state_t, jnp.asarray(g_keys))

        state_t = state_t.replace(step=state_t.step + 1)

        return jax.lax.cond(
            jnp.mod(state_t.step, config.log_every) == 0,
            _reset_metrics,
            identity,
            state_t,
        )

    return jax.lax.fori_loop(0, num_steps, _loop_step, state)


@functools.cache
def _make_metrics(config: Config) -> Type[clu_metrics.Collection]:
    metrics = dict(
        mmd=clu_metrics.Average.from_output("mmd"),
        observation=clu_metrics.Average.from_output("observation"),
    )

    if config.num_discriminator_steps > 0:
        return clu_metrics.Collection.create(**(metrics | dict(embedding=clu_metrics.Average.from_output("embedding"))))
    else:
        return clu_metrics.Collection.create(**metrics)


def _make_generator_state(
    rng: jax.random.KeyArray, observation_spec: specs.DiscreteArray, config: Config
) -> FittedValueTrainState:
    model = DistributionalSRGenerator(config.generator, config.num_outer, math.prod(observation_spec.shape))

    params = model.lazy_init(
        rng,
        jax.ShapeDtypeStruct(
            (
                config.num_outer,
                config.latent_dims + math.prod(observation_spec.shape),
            ),
            config.dtype,
        ),
    )
    params = typing.cast(PyTree[jax.Array], params)
    return FittedValueTrainState.create(
        apply_fn=model.apply,
        params=params,
        target_params_update=config.target_params_update,
        metrics=_make_metrics(config).empty(),
        tx=config.generator_optim,
    )


def _make_discriminator_state(
    rng: jax.random.KeyArray, observation_spec: specs.DiscreteArray, config: Config
) -> TrainState:
    model = DistributionalSRDiscriminator(
        config.discriminator,
        num_atoms=config.num_outer,
        with_separate_discriminator=config.inner_separate_discriminator,
    )  # type: ignore
    params_rng, spectral_norm_rng = jax.random.split(rng)

    variables = model.lazy_init(
        {"params": params_rng, "spectral_norm": spectral_norm_rng},
        jax.ShapeDtypeStruct(
            # We've vmapped over batch dimension and outer dimension
            # in the discriminator model
            (
                config.batch_size,
                config.num_outer,
                config.num_inner,
                math.prod(observation_spec.shape),
            ),
            config.dtype,
        ),
    )

    params = variables.pop("params")
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        variables=variables,
        metrics=_make_metrics(config).empty(),
        tx=config.discriminator_optim,
    )


def make_state(
    rng: jax.random.KeyArray,
    observation_spec: specs.DiscreteArray,
    config: Config,
) -> State:
    g_rng, d_rng = jax.random.split(rng)
    return State(
        step=jnp.int32(0),
        generator=_make_generator_state(g_rng, observation_spec, config),
        discriminator=_make_discriminator_state(d_rng, observation_spec, config),
    )


def load_state_and_config(
    checkpoint_path: pathlib.Path,
    checkpoint_step: int | None = None,
) -> tuple[FittedValueTrainState, Config]:
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_path,
        checkpointers={"generator": orbax.checkpoint.PyTreeCheckpointer()},
    )

    config: Config = fdl.build(serialization.load_json((checkpoint_path / "config.json").read_text()))
    env = stade.make(config.env)

    state = checkpoint_manager.restore(
        checkpoint_step or checkpoint_manager.latest_step(),  # pyright: ignore
        items={
            "generator": jax.eval_shape(
                functools.partial(_make_generator_state, config=config),
                jax.random.PRNGKey(0),
                env.observation_spec(),
            ),
        },
    )

    return state["generator"], config


def train(
    state: State,
    data: TransitionDataset,
    /,
    *,
    rng: jax.random.KeyArray,
    metrics_callback: Callable[[dict[str, jax.Array]], None] = lambda _: None,
    plot_callback: Callable[[State], None] = lambda _: None,
    checkpoint_callback: Callable[[State], None] = lambda _: None,
    config: Config,
) -> State:
    assert config.num_grad_updates % config.plot_every == 0
    if state.step >= config.num_grad_updates:
        logging.warning("Already trained for %d steps, did you restore from a checkpoint?", config.num_grad_updates)
        return state

    while (remaining_steps := config.num_grad_updates - state.step) > 0:
        rng, step_rng = jax.random.split(rng)
        state = train_loop(
            state,
            data,
            step_rng,
            num_steps=min(config.plot_every, remaining_steps),
            metrics_callback=metrics_callback,
            config=config,
        )
        plot_callback(state)
        checkpoint_callback(state)

    return state
