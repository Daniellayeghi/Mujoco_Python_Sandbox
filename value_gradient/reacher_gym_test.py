from reacher_gym import CustomReacher
import gym
import torch
gym.logger.MIN_LEVEL = gym.logger.min_level
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
env_name = 'CustomDoubleIntegrator'
nproc = 6
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('reward', self.training_env.get_attr('reward')[0])

        return True


from multiprocessing import  freeze_support
if __name__ == "__main__":

    freeze_support()
    def make_env(env_id, seed):
        def _f():
            env = CustomReacher(env_id, (-1, 1), 501)
            env.seed(seed)
            return env
        return _f

    envs= [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)

    model = PPO('MlpPolicy', envs,  verbose=1, device='cpu', tensorboard_log="./ppo_tensorboard_reach/")
    print(model.policy)
    a = model.get_parameters()
    model.learn(total_timesteps=1000000, log_interval=1, callback=TensorboardCallback())

    # Close the environments after training is complete
    envs.close()