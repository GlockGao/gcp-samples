import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


print('#' * 20)
print('1')

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_num_cpu_devices', 8)

arr = jnp.arange(32.0).reshape(4, 8)
arr.devices()
mesh = jax.make_mesh((2, 4), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
arr_sharded = jax.device_put(arr, sharding)

print('#' * 20)
print('2')

@jax.jit
def f_elementwise(x):
  return 2 * jnp.sin(x) + 1

result = f_elementwise(arr_sharded)

print('arr_sharded sharding:', arr_sharded.sharding)
print('result sharding:', result.sharding)
print("shardings match:", result.sharding == arr_sharded.sharding)

print('#' * 20)
print('3')

@jax.jit
def f_contract(x):
  return x.sum(axis=0)

result = f_contract(arr_sharded)
jax.debug.visualize_array_sharding(result)
print('arr_sharded:', arr_sharded)
print('result:', result)
