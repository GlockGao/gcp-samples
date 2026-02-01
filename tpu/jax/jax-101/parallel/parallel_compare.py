import jax
from jax.sharding import PartitionSpec as P
import numpy as np
from functools import partial

print('#' * 20)
print('1')

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_num_cpu_devices', 8)

@jax.jit
def layer(x, weights, bias):
  return jax.nn.sigmoid(x @ weights + bias)

rng = np.random.default_rng(0)

x = rng.normal(size=(32,))
weights = rng.normal(size=(32, 4))
bias = rng.normal(size=(4,))

result = layer(x, weights, bias)

print('x shape:', x.shape)
print('weights shape:', weights.shape)
print('bias shape:', bias.shape)
print('result shape:', result.shape)

print('x:\n', x)
print('weights:\n', weights)
print('bias:\n', bias)
print('result:\n', result)


print('#' * 20)
print('2')

mesh = jax.make_mesh((8,), ('x',))
x_sharded = jax.device_put(x, jax.NamedSharding(mesh, P('x')))
weights_sharded = jax.device_put(weights, jax.NamedSharding(mesh, P()))

result_sharded = layer(x_sharded, weights_sharded, bias)

print('x_sharded shape:', x_sharded.shape)
print('weights_sharded shape:', weights_sharded.shape)
print('bias shape:', bias.shape)
print('result_sharded shape:', result_sharded.shape)

print('x_sharded devices:', x_sharded.devices())
print('weights_sharded devices:', weights_sharded.devices())
print('result_sharded devices:', result_sharded.devices())

print('x_sharded sharding:', x_sharded.sharding)
print('weights_sharded sharding:', weights_sharded.sharding)
print('result_sharded sharding:', result_sharded.sharding)

print('#' * 20)
print('3')

from functools import partial

@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('x'), P('x', None), P(None)),
         out_specs=P(None))
def layer_sharded(x, weights, bias):
  return jax.nn.sigmoid(jax.lax.psum(x @ weights, 'x') + bias)

print(layer_sharded(x, weights, bias))