import time
from functools import partial

import jax
from jax import grad, jit, lax
import jax.numpy as jnp

print('#' * 20)
print('1')

@jit
def f(x):
  for i in range(3):
    x = 2 * x
  return x

print(f(3))

@jit
def g(x):
  y = 0.
  for i in range(x.shape[0]):
    y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))

print('#' * 20)
print('2')

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

f = jit(f, static_argnames='x')

print(f(2.))

def f(x, n):
  y = 0.
  print('n:', n)
  for i in range(n):
    print('i:', i, 'x[i]:', x[i])
    y = y + x[i]
  return y

f = jit(f, static_argnames='n')

print(f(jnp.array([2., 3., 4.]), 2))

print('#' * 20)
print('3')

def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)

operand = jnp.array([0.])
print(lax.cond(True, lambda x: x+1, lambda x: x-1, operand))
# --> array([1.], dtype=float32)
print(lax.cond(False, lambda x: x+1, lambda x: x-1, operand))
# --> array([-1.], dtype=float32)

print('#' * 20)
print('4')

def while_loop(cond_fun, body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val

init_val = 0
cond_fun = lambda x: x < 10
body_fun = lambda x: x+1
print(lax.while_loop(cond_fun, body_fun, init_val))
# --> array(10, dtype=int32)

print('#' * 20)
print('5')

def fori_loop(start, stop, body_fun, init_val):
  val = init_val
  for i in range(start, stop):
    val = body_fun(i, val)
  return val

init_val = 0
start = 0
stop = 10
body_fun = lambda i,x: x+i
print(lax.fori_loop(start, stop, body_fun, init_val))
# --> array(45, dtype=int32)  