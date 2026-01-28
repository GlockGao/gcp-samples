import jax
import jax.numpy as jnp


print('#' * 20)
print('1')

global_list = []

def log2(x):
    global_list.append(x)
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))

print('#' * 20)
print('2')

def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))

print('#' * 20)
print('3')

def log2_if_rank_2(x):
  if x.ndim == 2:
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.numpy.array([1, 2, 3]).ndim)
print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
print(jax.numpy.array([[1, 2, 3]]).ndim)
print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([[1, 2, 3]])))
