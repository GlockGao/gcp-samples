import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


print('#' * 20)
print('1')

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_num_cpu_devices', 8)

some_array = np.arange(8)
print(f"JAX-level type of some_array: {jax.typeof(some_array)}")

@jax.jit
def foo(x):
  print(f"JAX-level type of x during tracing: {jax.typeof(x)}")
  return x + x

print(foo(some_array))


print('#' * 20)
print('2')

from jax.sharding import AxisType

mesh = jax.make_mesh((2, 4), ("X", "Y"),
                     axis_types=(AxisType.Explicit, AxisType.Explicit))

replicated_array = np.arange(8).reshape(4, 2)
sharded_array = jax.device_put(replicated_array, jax.NamedSharding(mesh, P("X", None)))

print(f"replicated_array type: {jax.typeof(replicated_array)}")
print(f"sharded_array type: {jax.typeof(sharded_array)}")

print('#' * 20)
print('3')

arg0 = jax.device_put(np.arange(4).reshape(4, 1),
                      jax.NamedSharding(mesh, P("X", None)))
arg1 = jax.device_put(np.arange(8).reshape(1, 8),
                      jax.NamedSharding(mesh, P(None, "Y")))

print(f"arg0 type: {jax.typeof(arg0)}")
print(f"arg1 type: {jax.typeof(arg1)}")
print('arg0 devices:', arg0.devices())
print('arg1 devices:', arg1.devices())
print('arg0 sharding:', arg0.sharding)
print('arg1 sharding:', arg1.sharding)
print('arg0:\n', arg0)
print('arg1:\n', arg1)

@jax.jit
def add_arrays(x, y):
  ans = x + y
  print(f"x sharding: {jax.typeof(x)}")
  print(f"y sharding: {jax.typeof(y)}")
  print(f"ans sharding: {jax.typeof(ans)}")
  return ans

# with jax.set_mesh(mesh):
with mesh:
  print(add_arrays(arg0, arg1))
