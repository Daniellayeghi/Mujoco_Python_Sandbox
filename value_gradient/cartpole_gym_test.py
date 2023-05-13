import numpy as np
import gym
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from cartpole_gym import CustomCartPole

env_name = 'CustomCartpole'
nproc = 1
T=10
np.random.seed(0)

from multiprocessing import  freeze_support
if __name__ == "__main__":

    freeze_support()
    u = np.random.randn(10)
    x_test = np.load('x_test.npy')

    def make_env(env_id, seed):
        def _f():
            env = CustomCartPole(env_id, (-0.3, 0.3), 100)
            env.seed(seed)
            return env
        return _f

    envs= [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)

    Xt = envs.reset()
    xs = []
    for t in range(T):
        a = u[t].reshape(1, 1)
        xtpl, rt, done, info = envs.step(a)
        xs.append(xtpl)
    print("Done")