import jax
import jax.numpy as jnp
from jax import grad


print('#' * 20)
print('1')

grad_tanh = grad(jnp.tanh)
print(grad_tanh(2.0))