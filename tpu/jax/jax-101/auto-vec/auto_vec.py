import jax
import jax.numpy as jnp


print('#' * 20)
print('1')

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

auto_batch_convolve = jax.vmap(convolve)

print(xs.shape, ws.shape)  # (2, 5) (2, 3)
print(auto_batch_convolve(xs, ws))


print('#' * 20)
print('2')

auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

print(xst.shape, wst.shape)  # (5, 2) (3, 2)

print(auto_batch_convolve_v2(xst, wst).shape)  # (3, 2)
print(auto_batch_convolve_v2(xst, wst))


print('#' * 20)
print('3')

batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

print(batch_convolve_v3(xs, w))
print(batch_convolve_v3(xs, w).shape)  # (2, 3)


print('#' * 20)
print('4')

jitted_batch_convolve = jax.jit(auto_batch_convolve)

print(jitted_batch_convolve(xs, ws))
print(jitted_batch_convolve(xs, ws).shape)  # (2, 3)