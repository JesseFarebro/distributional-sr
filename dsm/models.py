import functools
from typing import Annotated, Any, Callable, Type

import flax.linen as nn
import jax
import jax.numpy as jnp

from dsm.tags import DType


class MLP(nn.Module):
    num_layers: int
    num_hidden_units: int
    num_outputs: int | None = None
    module: Type[nn.Module] = nn.Dense
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    dtype: Annotated[Any, DType] = jnp.float32
    param_dtype: Annotated[Any, DType] = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, *, num_outputs: int | None = None) -> jax.Array:
        num_outputs = nn.merge_param("num_outputs", self.num_outputs, num_outputs)
        for _ in range(self.num_layers):
            x = self.module(self.num_hidden_units, dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = self.activation(x)
        return self.module(num_outputs, dtype=self.dtype, param_dtype=self.param_dtype)(x)


class ResidualMLP(nn.Module):
    num_hidden_units: int
    num_layers_per_block: int
    num_blocks: int
    num_outputs: int | None = None
    module: Type[nn.Module] = nn.Dense
    activation: Callable[[jax.Array], jax.Array] = nn.leaky_relu
    dtype: Annotated[Any, DType] = jnp.float32
    param_dtype: Annotated[Any, DType] = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, *, num_outputs: int | None = None) -> jax.Array:
        num_outputs = nn.merge_param("num_outputs", self.num_outputs, num_outputs)
        PartialDense = functools.partial(self.module, dtype=self.dtype, param_dtype=self.param_dtype)

        x = PartialDense(self.num_hidden_units)(x)

        for _ in range(self.num_blocks):
            block_input = x

            x = self.activation(x)
            x = MLP(
                num_layers=self.num_layers_per_block,
                num_hidden_units=self.num_hidden_units,
                num_outputs=self.num_hidden_units,
                module=self.module,
                activation=self.activation,
            )(x)

            x += block_input

        x = self.activation(x)
        x = PartialDense(num_outputs)(x)
        return x
