from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from jax import jacfwd, jacrev
import jax.numpy as jnp
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')


def hessian(function):
    return jacfwd(jacrev(function))


def newtons_method(eval_func, input, eps, iter):
    delta = np.ones_like(input) * 1e10
    while np.linalg.norm(delta) > eps:
        hes = np.array(hessian(eval_func)(jnp.array(input))).reshape(input.shape[1], input.shape[1])
        jac = np.array(jacfwd(eval_func)(jnp.array(input))).reshape(input.shape[0], input.shape[1])
        delta = 1*(np.linalg.inv(hes).dot(jac.T)).T
        input -= delta
        iter += 1
        print(iter)
    return input


def f(x, y):
    return np.sin(np.sqrt(x[:, 1] ** 2 + y[:, 1] ** 2))


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()

