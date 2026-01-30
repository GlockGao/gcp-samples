import numpy as np
import jax
from jax import random


print('#' * 20)
print('1')

np.random.seed(0)

def bar(): return np.random.uniform()
def baz(): return np.random.uniform()

def foo(): return bar() + 2 * baz()

print(foo())

print('#' * 20)
print('2')

key = random.key(42)
print(key)

print(random.normal(key))
print(random.normal(key))

print('#' * 20)
print('3')

for i in range(3):
  new_key, subkey = random.split(key)
  del key  # The old key is consumed by split() -- we must never use it again.

  val = random.normal(subkey)
  del subkey  # The subkey is consumed by normal().

  print(f"draw {i}: {val}")
  key = new_key  # new_key is safe to use in the next iteration.

print('#' * 20)
print('4')

key, subkey = random.split(key)
print(key)
print(subkey)

key, *forty_two_subkeys = random.split(key, num=43)
print(key)
print(len(forty_two_subkeys))
print(*forty_two_subkeys, sep='\n')


print('#' * 20)
print('5')

key = random.key(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.key(42)
print("all at once :", random.normal(key, shape=(3,)))

print("vectorized  :", jax.vmap(random.normal)(subkeys))