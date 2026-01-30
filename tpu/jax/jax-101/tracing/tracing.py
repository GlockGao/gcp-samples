from jax import jit
import jax.numpy as jnp
import numpy as np


print('#' * 20)
print('1')

@jit
def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result

x = np.random.randn(3, 4)
y = np.random.randn(4)

print('x:', x)
print('x.shape:', x.shape)
print('y:', y)
print('y.shape:', y.shape)
print('First call to f():')
print(f(x, y))

print('#' * 20)
print('2')

x2 = np.random.randn(3, 4)
y2 = np.random.randn(4)
print('Second call to f():')
print(f(x2, y2))

print('#' * 20)
print('3')

from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

print(make_jaxpr(f)(x, y))

print('#' * 20)
print('4')
@jit
def f(x, neg):
  return -x if neg else x

try:
  print('Call to f(1, True):')
  print(f(1, True))
except Exception as e:
  print('Error:', e)

from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

print(f(1, True))
print(f(1, False))
