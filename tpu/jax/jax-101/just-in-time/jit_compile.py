import time
from functools import partial

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



@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))


print('#' * 20)
print('7')

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

start_time = time.time()

print("jit called in a loop with partials:")
g_inner_jitted_partial(10, 20).block_until_ready()

end_time = time.time()

avg_time = (end_time - start_time)

print(f"执行时间: {avg_time * 1000:.4f} ms")

start_time = time.time()

print("jit called in a loop with lambdas:")
g_inner_jitted_lambda(10, 20).block_until_ready()

end_time = time.time()

avg_time = (end_time - start_time)

print(f"执行时间: {avg_time * 1000:.4f} ms")

start_time = time.time()

print("jit called in a loop with caching:")
g_inner_jitted_normal(10, 20).block_until_ready()

end_time = time.time()

avg_time = (end_time - start_time)

print(f"执行时间: {avg_time * 1000:.4f} ms")
