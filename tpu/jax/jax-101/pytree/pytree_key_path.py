import collections
import jax

print('#' * 20)
print('1')

ATuple = collections.namedtuple("ATuple", ('name'))

tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

for key_path, value in flattened:
  print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')


print('#' * 20)
print('2')

for key_path, _ in flattened:
  print(f'Key path of tree{jax.tree_util.keystr(key_path)}: {repr(key_path)}')