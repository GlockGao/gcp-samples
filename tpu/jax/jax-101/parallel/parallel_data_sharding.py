import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


print('#' * 20)
print('1')

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_num_cpu_devices', 8)

print(jax.devices())

print('#' * 20)
print('2')

arr = jnp.arange(32.0).reshape(4, 8)
print('arr:', arr)
print('arr shape:', arr.shape)
print('arr devices:', arr.devices())
print('arr sharding:', arr.sharding)

jax.debug.visualize_array_sharding(arr)


print('#' * 20)
print('3')

mesh = jax.make_mesh((2, 4), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
print(sharding)

arr_sharded = jax.device_put(arr, sharding)

print(arr_sharded)
jax.debug.visualize_array_sharding(arr_sharded)
