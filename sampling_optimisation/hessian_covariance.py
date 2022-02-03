import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev
import pandas as pd
import cma
import matplotlib.pyplot as plt


def lee_func(input):
    return jnp.sin(input[:, 1]*10*jnp.pi)/(2*input[:, 1]) + (input[:, 1] - 1)**4


def function_eval_cma(input):
    return input[0]**4


def function_eval(input):
    return input[:, 0]**4


def eval_func(input):
    return (input[:, 0] + 2*input[:, 1] - 7)**2 + (2*input[:, 0] + input[:, 1] - 5)**2


def eval_func_cma(input):
    return (input[0] + 2*input[1] - 7)**2 + (2*input[0] + input[1] - 5)**2


def hessian(function):
    return jacfwd(jacrev(function))


def newtons_method(eval_func, input, eps, iter):
    delta = np.ones_like(input) * 1e10
    while np.linalg.norm(delta) > eps:
        hes = np.array(hessian(eval_func)(jnp.array(input))).reshape(input.shape[1], input.shape[1])
        hes = hes + np.identity(input.shape[1]) * 0.00001
        print(hes)
        jac = np.array(jacfwd(eval_func)(jnp.array(input))).reshape(input.shape[0], input.shape[1])
        delta = 1*(np.linalg.inv(hes).dot(jac.T)).T
        input -= delta
        iter += 1
        print(iter)
    return


if __name__ == "__main__":
    # Varying Hessian Example
    # Solve minimisation with Newton's method
    # x = np.array([[1.0]])
    # jac = jacfwd(function_eval)(jnp.array(x))
    # hes = hessian(function_eval)(jnp.array(x))
    # print(f"Jac: {jac}, Hes: {hes}")
    # result = newtons_method(function_eval, np.array([[1.0]]), 0.0001, 0)
    # print(f"Min: {result}")

    # Varying Hessian Example
    # Solve minimisation with Newton's method
    x = np.array([[2.0]])
    hes = hessian(lee_func)(jnp.array(x))
    print(f"Hessiain {hes}")
    result = newtons_method(lee_func, np.array([[2.0]]), 0.0001, 0)
    print(f"Min: {result}")

    # Solve minimisation with CMA
    x_cma = np.array([[100.0, 0.0]])
    es = cma.CMAEvolutionStrategy(x_cma, 9)
    es.optimize(function_eval_cma)
    cov = np.power(es.result.stds, 2)
    mean = es.result.xbest
    print(f"MEAN: {mean}, COV: {cov}")

    mean_csv = pd.read_csv('./outcmaes/xmean.dat', header=None, delim_whitespace=True, engine='python', skiprows=1)
    std_csv = pd.read_csv('./outcmaes/stddev.dat', header=None, delim_whitespace=True, engine='python', skiprows=1)

    std_mat = std_csv.to_numpy()
    mean_mat = mean_csv.to_numpy()
    mean_samples = np.delete(mean_mat, [0, 1, 2, 3, 4], 1)
    std_samples = np.delete(std_mat, [0, 1, 2, 3, 4], 1)

    # Compute the hessian of samples
    result = function_eval(mean_samples)
    samp_hessian = np.empty((mean_samples.shape[0], x.shape[1], x.shape[1]))

    for iter in range(mean_samples.shape[0]):
        mean = mean_samples[iter][0].reshape(1, x.shape[1])
        samp_hessian[iter] = np.array(hessian(function_eval)(jnp.array(mean))).reshape(x.shape[1], x.shape[1])

    # Plot the convergence of the mean, hessian of mean and variance
    iteration = np.linspace(0, mean_samples.shape[0]-1,  mean_samples.shape[0])
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Mean, Hessian of Mean and Covariance')
    ax1.plot(iteration, np.delete(mean_samples, 1, 1), '--r')
    ax2.plot(iteration, samp_hessian.reshape(mean_samples.shape[0], 1), 'b--')
    ax3.plot(iteration, np.delete(std_samples, 1, 1)**2, 'g--')
    plt.show()

    # Constant Hessian Example
    # Solve minimisation with newtons method
    x = np.array([[100.0, 100.0]])
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
    original_csv = pd.read_csv('./outcmaes/xmean.dat', header=None, delim_whitespace=True, engine='python', skiprows=1)
    csv_mat = original_csv.to_numpy()
    mean_samples = np.delete(csv_mat, [0, 1, 2, 3, 4], 1)

    # Compute the hessian of samples
    result = eval_func(mean_samples)
    samp_hessian = np.empty((mean_samples.shape[0], x.shape[1], x.shape[1]))

    for iter in range(mean_samples.shape[0]):
        mean = mean_samples[iter].reshape(1, x.shape[1])
        samp_hessian[iter] = np.array(hessian(eval_func)(jnp.array(x))).reshape(x.shape[1], x.shape[1])
