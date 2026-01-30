import jax.numpy as jnp
from jax import jit
import numpy as np


print('#' * 20)
print('1')

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())

x = jnp.ones((2, 3))
try:
  print('Call to f(x):')
  print(f(x))
except Exception as e:
  print('Error:', e)


print('#' * 20)
print('2')

@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prod())

try:
  print('Call to f(x):')
  f(x)
except Exception as e:
  print('Error:', e)

print('#' * 20)
print('3')

@jit
def f(x):
  return x.reshape((np.prod(x.shape),))

print(f(x))
