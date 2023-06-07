import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from RobotModelEnv_spot import RobotModelEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import results_plotter
import visdom
from visdom import Visdom
# from Visdom import VisdomCallback
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model_1")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


# ---------------- Create environment
# action_type can be set as discrete or continuous
env = RobotModelEnv(action_type='continuous')
# log_dir = r"C:\Users\Owner\Documents\shirelle\RobotDog\saved_models\25.04.23_try_1"
# env = Monitor(env, log_dir)

'''load the model and predict'''
# Option 2: load the model from files (note that the loaded model can be learned again)
print("load the model from files")
model = SAC.load(r"C:\Users\Owner\Documents\shirelle\RobotDog\saved_models\18.05.23_try_2\best_model.zip", env=env)

print('Prediction')

rewards = []
cum_rewards = []
gamma = 0.99
cumulative_reward = 0
i = 0

observation, done = env.reset(), False
episode_reward = 0.0

while not done and i < 105:
    action, _state = model.predict(observation, deterministic=True)
    print("-------- action number {} ------------".format(i))
    print("action = {}".format(action))
    observation, reward, done, info = env.step(action)
    print("reward = {}".format(reward))
    cumulative_reward = reward + gamma * cumulative_reward
    print("cumulative_reward = {}".format(cumulative_reward))
    rewards.append(reward)
    cum_rewards.append(cumulative_reward)
    if done:
        env.reset()
        done = False
        i += 1

    # print([episode_reward, env.counts])
print("rewards = {}".format(rewards))
print("cum rewards = {}".format(cum_rewards))
timesteps = list(range(len(rewards)))
plt.plot(timesteps, cum_rewards)
plt.show()
# env.close()
