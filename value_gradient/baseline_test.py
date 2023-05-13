import gym
gym.logger.MIN_LEVEL = gym.logger.min_level
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _init

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('reward', self.training_env.get_attr('reward')[0])

        return True


from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()

    # Create 4 environments and run them in parallel using 4 processes
    env = gym.make('Pendulum-v1')
    num_envs = 8
    envs = [make_env('Pendulum-v1', i) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    model = PPO(
        'MlpPolicy', envs, verbose=1, tensorboard_log="./ppo_tensorboard_pend/",
        n_steps=1024, learning_rate=1e-3, gamma=0.9,
        gae_lambda=0.95, sde_sample_freq=4, use_sde=True
    )

    model.learn(total_timesteps=200000, callback=TensorboardCallback())

    # Close the environments after training is complete
    envs.close()