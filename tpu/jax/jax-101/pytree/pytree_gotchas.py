import jax
import jax.numpy as jnp


print('#' * 20)
print('1')

a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

# Try to make another pytree with ones instead of zeros.
shapes = jax.tree.map(lambda x: x.shape, a_tree)
print('shapes:', shapes)
print(jax.tree.map(jnp.ones, shapes))

print('#' * 20)
print('2')

print(jax.tree.leaves([None, None, None]))
print(jax.tree.leaves([None, None, None], is_leaf=lambda x: x is None))
