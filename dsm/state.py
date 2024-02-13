import dataclasses
import typing
from typing import Protocol

import jax
import optax
from clu import metrics as clu_metrics
from flax import struct
from flax.training import train_state
from typing_extensions import Self

from dsm.types import PyTree


class TrainState(train_state.TrainState):
    variables: PyTree[jax.Array]
    metrics: clu_metrics.Collection = struct.field(default_factory=clu_metrics.Collection.create_collection)  # pyright: ignore


@typing.runtime_checkable
class TargetParamsUpdate(Protocol):
    def __call__(
        self, *, old_params: PyTree[jax.Array], new_params: PyTree[jax.Array], steps: int
    ) -> PyTree[jax.Array]: ...


@dataclasses.dataclass(frozen=True)
class HardTargetParamsUpdate:
    update_period: int

    def __call__(
        self, *, old_params: PyTree[jax.Array], new_params: PyTree[jax.Array], steps: int
    ) -> PyTree[jax.Array]:
        return optax.periodic_update(
            new_tensors=new_params,
            old_tensors=old_params,
            steps=steps,  # pyright: ignore
            update_period=self.update_period,
        )


@dataclasses.dataclass(frozen=True)
class SoftTargetParamsUpdate:
    step_size: float

    def __call__(
        self, *, old_params: PyTree[jax.Array], new_params: PyTree[jax.Array], steps: int
    ) -> PyTree[jax.Array]:
        del steps
        return optax.incremental_update(  # pyright: ignore
            new_tensors=new_params,
            old_tensors=old_params,
            step_size=self.step_size,
        )


class FittedValueTrainState(train_state.TrainState):
    target_params: PyTree[jax.Array]  # pyright: ignore
    target_params_update: TargetParamsUpdate = struct.field(pytree_node=False)  # pyright: ignore
    metrics: clu_metrics.Collection = struct.field(default_factory=clu_metrics.Collection.create_collection)  # pyright: ignore

    def apply_gradients(
        self,
        /,
        grads: optax.Updates,
        **kwargs,
    ) -> Self:
        state = super().apply_gradients(grads=grads)
        new_target_params = self.target_params_update(
            new_params=typing.cast(PyTree[jax.Array], state.params),
            old_params=state.target_params,
            steps=state.step,
        )

        return state.replace(target_params=new_target_params, **kwargs)

    @classmethod
    def create(
        cls,
        /,
        *,
        params: PyTree[jax.Array],
        target_params_update: TargetParamsUpdate,
        **kwargs,
    ) -> Self:
        target_params = jax.tree_util.tree_map(lambda x: x, params)
        return super().create(
            params=params,
            target_params=target_params,
            target_params_update=target_params_update,
            **kwargs,
        )


class State(struct.PyTreeNode):
    step: jax.Array
    generator: FittedValueTrainState
    discriminator: TrainState
