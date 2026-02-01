from jax import jit
import jax.numpy as jnp


def python_check_positive_even(x):
  is_even = x % 2 == 0
  # `and` short-circults, so when `is_even` is `False`, `x > 0` is not evaluated.
  return is_even and (x > 0)

@jit
def jax_check_positive_even(x):
  is_even = x % 2 == 0
  # `logical_and` does not short circuit, so `x > 0` is always evaluated.
  return jnp.logical_and(is_even, x > 0)

print(python_check_positive_even(24))
print(jax_check_positive_even(24))

x = jnp.array([-1, 2, 5])
print(jax_check_positive_even(x))