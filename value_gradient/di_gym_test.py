import numpy as np
from utilities.torch_device import device
import gym
gym.logger.MIN_LEVEL = gym.logger.min_level
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from di_gym import CustomDoubleIntegrator
from stable_baselines3.common.callbacks import BaseCallback
env_name = 'CustomDoubleIntegrator'
nproc = 1
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('mean_reward', self.training_env.get_attr('mean_reward')[0])

        return True

import torch
from basline_net_utils import ValueNet, extract_value_net
from multiprocessing import freeze_support
if __name__ == "__main__":

    freeze_support()
    def make_env(env_id, seed):
        env = CustomDoubleIntegrator(env_id, (-3, 3), 101)
        def _f():
            env.seed(seed)
            return env
        return _f

    envs= [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)

    model = PPO(
        'MlpPolicy', envs, verbose=1, tensorboard_log="./ppo_tensorboard_di/",
        n_steps=512, learning_rate=1e-3, gamma=0.9, batch_size=32, clip_range=0.3,
        n_epochs=5, gae_lambda=1, use_sde=True, normalize_advantage=True,
        max_grad_norm=0, ent_coef=7.52585e-08, vf_coef=0.95
    )

    model.learn(total_timesteps=500000, log_interval=1, callback=TensorboardCallback())
    v = ValueNet(*extract_value_net(model)).to(device)
    torch.save(v.state_dict(), 'baseline_value_net_di.pt')
    torch.save(v.cpu().state_dict(), 'baseline_value_net_di_cpu.pt')

    # Close the environments after training is complete
    envs.close()