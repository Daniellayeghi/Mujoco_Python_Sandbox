import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev
import pandas as pd
import cma


def eval_func(input):
    return (input[:, 0] + 2*input[:, 1] - 7)**2 + (2*input[:, 0] + input[:, 1] - 5)**2


def eval_func_cma(input):
    return (input[0] + 2*input[1] - 7)**2 + (2*input[0] + input[1] - 5)**2


def hessian(function):
    return jacfwd(jacrev(function))


def newtons_method(eval_func, input, eps, iter):
    delta = np.ones_like(input) * 1e10
    while np.linalg.norm(delta) > 1e-5:
        hes = np.array(hessian(eval_func)(jnp.array(input))).reshape(input.shape[1], input.shape[1])
        jac = np.array(jacfwd(eval_func)(jnp.array(input))).reshape(input.shape[0], input.shape[1])
        delta = (np.linalg.inv(hes).dot(jac.T)).T
        input -= delta
        iter += 1
    return input


if __name__ == "__main__":

    # Solve minimisation with newtons method
    input = np.array([[100.0, 100.0]])
    jac = jacfwd(eval_func)(jnp.array([[0.0], [0.0]]))
    hes = hessian(eval_func)(jnp.array([[1.0, 3.0]]))
    result = newtons_method(eval_func, np.array([[100.0, 100.0]]), 0.0001, 0)
    print(f"Min: {result}")

    # Solve minimisation with CMA
    es = cma.CMAEvolutionStrategy(np.array([[9.0, 8.0]]), 9)
    es.optimize(eval_func_cma)
    cov = np.power(es.result.stds, 2)
    mean = es.result.xbest
    print(f"MEAN: {mean}, COV: {cov}")

    # Load samples
    original_csv = pd.read_csv('./outcmaes/xmean.dat',
            header=None, delim_whitespace=True, engine='python', skiprows=1)
    csv_mat = original_csv.to_numpy()
    mean_samples = np.delete(csv_mat, [0, 1, 2, 3, 4], 1)

    # Compute the hessian of samples
    result = eval_func(mean_samples)
    samp_hessian = np.empty((mean_samples.shape[0], input.shape[1], input.shape[1]))

    for iter in range(mean_samples.shape[0]):
        mean = mean_samples[iter].reshape(1, input.shape[1])
        samp_hessian[iter] = np.array(hessian(eval_func)(jnp.array(input))).reshape(input.shape[1], input.shape[1])
