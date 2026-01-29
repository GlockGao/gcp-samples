import jax
import jax.numpy as jnp
from jax import grad


print('#' * 20)
print('1')

grad_tanh = grad(jnp.tanh)
print(grad_tanh(2.0))
print(grad(grad(jnp.tanh))(2.0))
print(grad(grad(grad(jnp.tanh)))(2.0))


print('#' * 20)
print('2')

f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)

print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))
