import jax
import jax.numpy as jnp

print('#' * 20)
print('1')

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

print(convolve(x, w))


print('#' * 20)
print('2')

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
print(xs.shape, ws.shape)  # (2, 5) (2, 3)

def manually_batched_convolve(xs, ws):
  output = []
  for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

print(manually_batched_convolve(xs, ws))  # (2, 3)


print('#' * 20)
print('3')
def manually_vectorized_convolve(xs, ws):
  output = []
  for i in range(1, xs.shape[-1] -1):
    output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

print(manually_vectorized_convolve(xs, ws))
