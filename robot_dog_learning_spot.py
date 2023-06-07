from stable_baselines3 import SAC
import os
import csv
import visdom
# from Visdom import VisdomCallback
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from RobotModelEnv_spot import RobotModelEnv

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
        self.save_path = os.path.join(log_dir, "best_model")
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


log_dir = r"C:\Users\Owner\Documents\shirelle\RobotDog\saved_models\random"
#
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir, exist_ok=True)
# ---------------- Create environment
env = RobotModelEnv(action_type='continuous')
env = Monitor(env, log_dir)

# '''new model + train'''
print("create a new model")
policy_kwargs = dict(net_arch=[512, 512, 512, 512])
model = SAC(policy="MlpPolicy", env=env, learning_rate=0.0003, verbose=True, gamma=0.99, batch_size=512,
            policy_kwargs=policy_kwargs)


checkpoint_callback = CheckpointCallback(save_freq=500, save_path=log_dir)
callback_best_reward = SaveOnBestTrainingRewardCallback(check_freq=250, log_dir=log_dir)

callback = CallbackList([checkpoint_callback, callback_best_reward])

model.learn(total_timesteps=1e6, callback=callback, progress_bar=True)

# print('---------------------------------------------------Finished--------------------------------------------------')

'''load the model and predict'''
# Option 2: load the model from files (note that the loaded model can be learned again)
# print("load the model from files")
# path = r"C:\Users\Owner\Documents\shirelle\RobotDog\saved_models\08.05.23_try_1\rl_model_23500_steps.zip"
# model = SAC.load(path=path, env=env)
# model.learn(total_timesteps=1e6, callback=callback, progress_bar=True)

'''plot the reward per episode as function of time'''
# plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "SAC")
# plt.show()

# print('Prediction')
#
# for _ in range(100):
#     observation, done = env.reset(), False
#     episode_reward = 0.0
#
#     while not done:
#         action, _state = model.predict(observation, deterministic=True)
#         observation, reward, done, info = env.step(action)
#         if done:
#             env.reset()
#         episode_reward += reward
#
#     print([episode_reward, env.counts])
#
# env.close()


# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(5), sigma=float(0.7) * np.ones(12))
# model = DDPG(policy='MlpPolicy', env=env, learning_rate=1e-3, verbose=True, action_noise=action_noise, gamma=0.95)

# vis.line(Y=model.predict(np.array([[0, 0, 0]])), X=np.array([1]), opts={'title': 'Predicted Value'})

# del model # delete the model and load the best model to predict
# model = A2C.load("../CartPole/saved_models/tmp/best_model", env=env)
