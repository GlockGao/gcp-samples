import time
import jax
import jax.numpy as jnp


print('#' * 20)
print('1')

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


x = jnp.arange(1000000)

start_time = time.time()

selu(x).block_until_ready()

end_time = time.time()

avg_time = (end_time - start_time)
print(f"执行时间: {avg_time * 1000:.4f} ms")


print('#' * 20)
print('2')

selu_jit = jax.jit(selu)

# Pre-compile the function before timing...
selu_jit(x).block_until_ready()

start_time = time.time()

selu_jit(x).block_until_ready()

end_time = time.time()

avg_time = (end_time - start_time)
print(f"执行时间: {avg_time * 1000:.4f} ms")


# Condition on value of x.
print('#' * 20)
print('3')

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

try:
    jax.jit(f)(10)  # Raises an error
except Exception as e:
    print("Error:", e)

print('#' * 20)
print('4')

# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

try:
    jax.jit(g)(10, 20)  # Raises an error
except Exception as e:
    print("Error:", e)


# While loop conditioned on x and n with a jitted body.
print('#' * 20)
print('5')

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

x = g_inner_jitted(10, 20)
print("Result:", x)


print('#' * 20)
print('6')

f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))

g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))

from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))