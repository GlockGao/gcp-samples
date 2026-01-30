import jax


print('#' * 20)
print('1')

def tree_transpose(list_of_trees):
  """
  Converts a list of trees of identical structure into a single tree of lists.
  """
  return jax.tree.map(lambda *xs: list(xs), *list_of_trees)

# Convert a dataset from row-major to column-major.
episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
print(tree_transpose(episode_steps))

print('#' * 20)
print('2')

print(jax.tree.transpose(
  outer_treedef = jax.tree.structure([0 for e in episode_steps]),
  inner_treedef = jax.tree.structure(episode_steps[0]),
  pytree_to_transpose = episode_steps
))