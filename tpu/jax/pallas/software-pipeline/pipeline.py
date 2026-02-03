import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np


print('#' * 20)
print('1')

# Note: This is a TPU example.

def add_matrices_kernel(x_sram_ref, y_sram_ref, z_sram_ref):
  # Load x and y from SRAM into registers
  x_regs = x_sram_ref[:, :]
  y_regs = y_sram_ref[:, :]
  # Execute a vectorized add
  z_regs = x_regs + y_regs
  # Store the output values in registers back into SRAM
  z_sram_ref[:, :] = z_regs


def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
  # pallas_call will first allocate scratch buffers for `x` and `y` in SRAM.
  # It will then copy `x` and `y` from HBM into SRAM.
  z = pl.pallas_call(
      add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
  # pallas_call will also copy the output from SRAM back into HBM.
  return z


x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
z = add_matrices(x, y)

print('x.shape:', x.shape)
print('x.dtype:', x.dtype)
print('y.shape:', y.shape)
print('y.dtype:', y.dtype)
print('z.shape:', z.shape)
print('z.dtype:', z.dtype)