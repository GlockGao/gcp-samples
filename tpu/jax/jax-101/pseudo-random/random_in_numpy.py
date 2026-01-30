import numpy as np


print('#' * 20)
print('1')

np.random.seed(0)

print(np.random.random())
print(np.random.random())
print(np.random.random())

print('#' * 20)
print('2')

def print_truncated_random_state():
  """To avoid spamming the outputs, print only part of the state."""
  full_random_state = np.random.get_state()
  print(str(full_random_state)[:460], '...')

print_truncated_random_state()

np.random.seed(0)
print_truncated_random_state()

_ = np.random.uniform()
print_truncated_random_state()

print('#' * 20)
print('3')

np.random.seed(0)
print(np.random.uniform(size=3))

np.random.seed(0)
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once :", np.random.uniform(size=3))
