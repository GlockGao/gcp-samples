from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

print('#' * 20)
print('1')

def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y

    print('o_ref:', o_ref)
    print('o_ref.dtype:', o_ref.dtype)
    print('o_ref.shape:', o_ref.shape)
    print('x_ref:', x_ref)
    print('x_ref.dtype:', x_ref.dtype)
    print('x_ref.shape:', x_ref.shape)
    print('y_ref:', y_ref)
    print('y_ref.dtype:', y_ref.dtype)
    print('y_ref.shape:', y_ref.shape)


def add_sliced_kernel(x_ref, y_ref, o_ref):
    small_mid = x_ref.shape[0] // 2

    x_left = x_ref.at[:small_mid]
    x_right = x_ref.at[small_mid:]
    y_left = y_ref.at[:small_mid]
    y_right = y_ref.at[small_mid:]

    # The output shape is (4*small_mid).
    large_mid = 2*small_mid
    o_ref.at[:large_mid][:small_mid] = x_left[...] + y_left[...]
    o_ref.at[:large_mid][small_mid:] = x_left[...] + y_right[...]
    o_ref.at[large_mid:][:small_mid] = x_right[...] + y_left[...]
    o_ref.at[large_mid:][small_mid:] = x_right[...] + y_right[...]

    print('o_ref:', o_ref)
    print('o_ref.shape:', o_ref.shape)
    print('large_mid:', large_mid)
    print('small_mid:', small_mid)


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)

print(add_vectors(jnp.arange(8), jnp.arange(8)))

print('#' * 20)
print('2')

def iota_kernel(o_ref):
    i = pl.program_id(0)
    o_ref[i] = i

    print('o_ref:', o_ref)
    print('o_ref.dtype:', o_ref.dtype)
    print('o_ref.shape:', o_ref.shape)
    print('i:', i)

# TPU version
from jax.experimental.pallas import tpu as pltpu

def iota(size: int):
    return pl.pallas_call(iota_kernel,
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
        grid=(size,))()

print(iota(8))


print('#' * 20)
print('3')

def matmul_kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] @ y_ref[...]

def matmul(x: jax.Array, y: jax.Array):
  return pl.pallas_call(
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
    grid=(2, 2),
    in_specs=[
        pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
    ],
    out_specs=pl.BlockSpec(
        (x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j),
    )
  )(x, y)
k1, k2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(k1, (1024, 1024))
y = jax.random.normal(k2, (1024, 1024))

z = matmul(x, y)

print('x.shape:', x.shape)
print('y.shape:', y.shape)
print('z.shape:', z.shape)

np.testing.assert_allclose(z, x @ y)