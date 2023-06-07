import os
import sys
import time
import csv
import gym
import math
import numpy as np
from gym import spaces
from gym.utils import seeding
import visdom
from timeit import default_timer

sys.path.append(
    os.path.abspath("C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python"))
from zmqRemoteApi import RemoteAPIClient

global client, sim, spot, legBase, base, target, spot_script, force_sensors, episode

client = RemoteAPIClient()
# Connect to VREP (CoppeliaSim)
print('Connect to CoppeliaSim')

client = RemoteAPIClient()
sim = client.getObject('sim')
spot = sim.getObject('/spot')
target = sim.getObject('/target')
legBase = sim.getObject('/legBase')
base = sim.getObject('/base')
tips = np.array([sim.getObject('./tip_FL'),
                 sim.getObject('./tip_FR'),
                 sim.getObject('./tip_BL'),
                 sim.getObject('./tip_BR')])

force_sensors = np.array([sim.getObject('/spot_front_left_lower_leg_force_sensor'),
                          sim.getObject('/spot_front_right_lower_leg_force_sensor'),
                          sim.getObject('/spot_rear_left_lower_leg_force_sensor'),
                          sim.getObject('/spot_rear_right_lower_leg_force_sensor')])
spot_script = sim.getScript(1, spot, '/spot')

log_dir = r"C:\Users\Owner\Documents\shirelle\RobotDog\saved_models\random"

if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

csv_file = r'C:\Users\Owner\Documents\shirelle\RobotDog\saved_models\random\train_data.csv'
csv_fields = ['step', 'reward', 'fall', 'passed_stair', 'target reach']
gamma = 0.99
vis = visdom.Visdom()
episode = 0
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)


class RobotModelEnv(gym.Env):
    # gym.Env.max_episode_steps = 200
    """Custom Environment that follows gym interface"""

    def __init__(self, action_type='discrete'):
        super(RobotModelEnv, self).__init__()
        self.csv_row = 0
        # create CSV file with header row
        self.passed_stair_flag = False
        self.reach_target_flag = False

        self.body_position = np.array(sim.getObjectPosition(spot, -1))
        tip_FL_pos = sim.getObjectPosition(sim.getObject('./tip_FL'), -1)
        tip_FR_pos = sim.getObjectPosition(sim.getObject('./tip_FR'), -1)
        tip_BL_pos = sim.getObjectPosition(sim.getObject('./tip_BL'), -1)
        tip_BR_pos = sim.getObjectPosition(sim.getObject('./tip_BR'), -1)
        self.leg_tip_positions_world_frame = np.concatenate((tip_FL_pos, tip_FR_pos, tip_BL_pos, tip_BR_pos))
        self.back_leg_position = min(self.leg_tip_positions_world_frame[0], self.leg_tip_positions_world_frame[3],
                                     self.leg_tip_positions_world_frame[6], self.leg_tip_positions_world_frame[9])
        self.average_tips_x_position = (self.leg_tip_positions_world_frame[0] +
                                        self.leg_tip_positions_world_frame[3] +
                                        self.leg_tip_positions_world_frame[6] +
                                        self.leg_tip_positions_world_frame[9]) / 4
        res, force_FL, torque_FL = sim.readForceSensor(sim.getObject('/spot_front_left_lower_leg_force_sensor'))
        res, force_FR, torque_FR = sim.readForceSensor(sim.getObject('/spot_front_right_lower_leg_force_sensor'))
        res, force_BL, torque_BL = sim.readForceSensor(sim.getObject('/spot_rear_left_lower_leg_force_sensor'))
        res, force_BR, torque_BR = sim.readForceSensor(sim.getObject('/spot_rear_right_lower_leg_force_sensor'))

        self.leg_force_sensors = np.concatenate(
            (force_FL, force_FR, force_BL, force_BR), axis=None, dtype=np.float32)

        self.target = np.array(sim.getObjectPosition(target, -1))
        self.data = np.concatenate((self.average_tips_x_position, self.body_position, self.leg_force_sensors,
                                    self.leg_tip_positions_world_frame, self.target), axis=None, dtype=np.float32)

        high_position = np.array([2, 1, 0.7], dtype=np.float32)
        low_position = np.array([-2, -1, -0.01], dtype=np.float32)

        high_x_position = np.array([2.0], dtype=np.float32)
        low_x_position = np.array([-2.0], dtype=np.float32)

        high_force_sensors = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
                                      dtype=np.float32)
        low_force_sensors = np.array(
            [-500, -500, -500, -500, -500, -500, -500, -500, -500, -500, -500, -500], dtype=np.float32)
        low_tip_positions = low_force_sensors
        high_tip_positions = high_force_sensors

        observation_space_min = np.concatenate(
            (low_x_position, low_position, low_force_sensors, low_tip_positions,
             low_position), axis=None,
            dtype=np.float32)
        observation_space_max = np.concatenate(
            (high_x_position, high_position, high_force_sensors,
             high_tip_positions, high_position), axis=None,
            dtype=np.float32)

        # action space - dx, dh, leg
        high_action = np.array([0.2, 0.05, 0.2, 4.999], dtype=np.float32)
        low_action = np.array([0.05, -0.05, 0.1, 1], dtype=np.float32)

        self.action_space = spaces.Box(low=low_action, high=high_action, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_space_min, high=observation_space_max, shape=(31,),
                                            dtype=np.float32)
        self.cum_reward = np.array([0])
        self.seed()
        self.counts = 0
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        global episode
        dx = float(action[0])
        dy = float(action[1])
        h = float(action[2])
        leg = int(action[3])
        print("dx = {}, dy = {}, h = {}, leg id = {}".format(dx, dy, h, leg))

        res = sim.callScriptFunction('step', spot_script, dx, dy, h, leg)
        signal = sim.getInt32Signal("execDone")
        fall_signal = sim.getStringSignal("fall")

        start = default_timer()

        while signal is None:
            duration = default_timer() - start
            signal = sim.getInt32Signal("execDone")
            fall_signal = sim.getStringSignal("fall")
            if duration > 15:
                fall_signal = True
            if fall_signal:
                break

        sim.clearInt32Signal("execDone")
        res, force_FL, torque_FL = sim.readForceSensor(sim.getObject('/spot_front_left_lower_leg_force_sensor'))
        res, force_FR, torque_FR = sim.readForceSensor(sim.getObject('/spot_front_right_lower_leg_force_sensor'))
        res, force_BL, torque_BL = sim.readForceSensor(sim.getObject('/spot_rear_left_lower_leg_force_sensor'))
        res, force_BR, torque_BR = sim.readForceSensor(sim.getObject('/spot_rear_right_lower_leg_force_sensor'))
        self.leg_force_sensors = np.concatenate(
            (force_FL, force_FR, force_BL, force_BR), axis=None, dtype=np.float32)
        normal_co = np.max(np.abs(self.leg_force_sensors))
        self.leg_force_sensors = self.leg_force_sensors / normal_co

        self.body_position = sim.getObjectPosition(spot, -1)

        tip_FL_pos = sim.getObjectPosition(sim.getObject('./tip_FL'), -1)
        tip_FR_pos = sim.getObjectPosition(sim.getObject('./tip_FR'), -1)
        tip_BL_pos = sim.getObjectPosition(sim.getObject('./tip_BL'), -1)
        tip_BR_pos = sim.getObjectPosition(sim.getObject('./tip_BR'), -1)
        self.leg_tip_positions_world_frame = np.concatenate((tip_FL_pos, tip_FR_pos, tip_BL_pos, tip_BR_pos))
        average_tips_x_position_old = self.average_tips_x_position
        self.average_tips_x_position = (self.leg_tip_positions_world_frame[0] +
                                        self.leg_tip_positions_world_frame[3] +
                                        self.leg_tip_positions_world_frame[6] +
                                        self.leg_tip_positions_world_frame[9]) / 4

        self.data = np.concatenate(
            (self.average_tips_x_position, self.body_position, self.leg_force_sensors,
             self.leg_tip_positions_world_frame, self.target),
            axis=None, dtype=np.float32)

        self.counts += 1
        self.csv_row += 1
        fall = bool(fall_signal)
        # limit to 50 actions
        if self.counts > 50:
            fall = True

        # if all legs passed the stair
        self.back_leg_position = min(self.leg_tip_positions_world_frame[0], self.leg_tip_positions_world_frame[3],
                                     self.leg_tip_positions_world_frame[6], self.leg_tip_positions_world_frame[9])

        reward = self.reward_function_18_05_try_2(leg, fall, average_tips_x_position_old)

        # plotting and saving data
        self.cum_reward = np.append(self.cum_reward, self.cum_reward[-1] + reward * gamma)
        if fall or self.reach_target_flag or self.passed_stair_flag:
            done = True
            # '''plot the cum reward as function of step once in 250 episodes'''
            # x = list(range(len(self.cum_reward)))
            # y = self.cum_reward
            # opts = {'title': 'Cumulative Rewards', 'xlabel': 'Step', 'ylabel': 'Cumulative Reward'}
            #
            # if episode % 50 == 0:
            #     vis.line(X=x, Y=y, opts=opts, win='cum_rewards_19.05.23_try_1', update='append', name='%s' % episode)

            '''save data in csv file'''
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.csv_row, self.cum_reward[-1], fall, self.passed_stair_flag, self.reach_target_flag])
                # if episode % 25 == 0:
                #     vis.scatter(X=[episode], Y=[self.cum_reward[-1]], name='%s' % episode, update='append',
                #                 win='reward_per_episode_19.05.23_try_1', opts=dict(markercolor=np.array([[0, 0, 1]]),
                #                                                                    markersize=5, xlabel='Episode',
                #                                                                    ylabel='Reward',
                #                                                                    title='Reward per episode'))
            print(self.cum_reward)
            episode += 1
        else:
            done = False

        print("end of step")
        print("reward = " + str(reward))
        return self.data, reward, done, {}

    def reset(self):
        self.counts = 0
        sim.stopSimulation()
        time.sleep(5)
        sim.startSimulation()
        time.sleep(5)
        self.__init__()
        return self.data

    def render(self):
        return None

    def close(self):
        sim.stopSimulation()  # stop the simulation
        print('Close the environment')
        return None

    # reward על התקדמות במקום על מיקום
    def reward_function_19_05_23_try_1(self, leg, fall, average_tips_x_position_old):
        delta_x = self.average_tips_x_position - average_tips_x_position_old
        print("delta_x = {}".format(delta_x))
        delta_x_reward = np.exp(2*delta_x) - 1
        reward = delta_x_reward
        print("reward = {}".format(reward))
        y_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
            print("received y_reward = {}".format(y_reward))
            reward += y_reward
        # robot passed the stair
        passed_stair_reward = 0
        if self.back_leg_position > 0.540 and self.passed_stair_flag == False:
            self.passed_stair_flag = True
            passed_stair_reward = 100 / self.counts
            reward += passed_stair_reward
            print("passed_stair_reward is received = {}".format(passed_stair_reward))
            print("self.counts = {}".format(self.counts))
        # reach target
        reach_target_reward = 0
        if self.back_leg_position > self.target[0] and self.reach_target_flag == False:
            self.reach_target_flag = True
            reach_target_reward = 400 / self.counts
            reward += reach_target_reward
            print("reach_target_reward is received = {}".format(reach_target_reward))
            print("self.counts = {}".format(self.counts))
        falling_reward = 0
        if fall:
            falling_reward = - 1
            print("falling reward = {}".format(falling_reward))
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}"
              .format(reach_target_reward, passed_stair_reward, falling_reward, delta_x_reward, y_reward))
        return reward

    def reward_function_18_05_try_2(self, leg, fall, average_tips_x_position_old):
        x_reward = (10 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5))
        reward = x_reward
        # if the robot stay in place he doesn't get the x reward
        not_moving_forward_reward = 0
        if average_tips_x_position_old - self.average_tips_x_position > - 0.015:
            not_moving_forward_reward = - 0.1
            reward = not_moving_forward_reward
        y_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
            reward += y_reward

        # robot passed the stair
        passed_stair_reward = 0
        if self.back_leg_position > 0.540 and self.passed_stair_flag == False:
            self.passed_stair_flag = True
            passed_stair_reward = 200 / self.counts
            reward = passed_stair_reward
        # reach target
        reach_target_reward = 0
        if self.back_leg_position > self.target[0] and self.reach_target_flag == False:
            self.reach_target_flag = True
            reach_target_reward = 400 / self.counts
            reward = reach_target_reward
        falling_reward = 0
        if fall:
            falling_reward = -0.3
            reward += falling_reward

        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_18_05_try_1(self, leg, fall, average_tips_x_position_old):
        x_reward = (10 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5))
        reward = x_reward
        # if the robot stay in place he doesn't get the x reward
        not_moving_forward_reward = 0
        if average_tips_x_position_old - self.average_tips_x_position > - 0.015:
            not_moving_forward_reward = - 0.1
            reward = not_moving_forward_reward
        y_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
            reward += y_reward

        # robot passed the stair
        passed_stair_reward = 0
        if self.back_leg_position > 0.540 and self.passed_stair_flag == False:
            self.passed_stair_flag = True
            passed_stair_reward = 200 / self.counts
            reward = passed_stair_reward
        # reach target
        reach_target_reward = 0
        if self.back_leg_position > self.target[0] and self.reach_target_flag == False:
            self.reach_target_flag = True
            reach_target_reward = 400 / self.counts
            reward = reach_target_reward
        falling_reward = 0
        if fall:
            falling_reward = -0.3
            reward += falling_reward

        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_14_05_try_1(self, leg, fall, average_tips_x_position_old):
        x_reward = (10 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5))
        reward = x_reward
        # if the robot stay in place he doesn't get the x reward
        not_moving_forward_reward = 0
        if average_tips_x_position_old - self.average_tips_x_position > - 0.015:
            not_moving_forward_reward = - 0.1
            reward = not_moving_forward_reward
        y_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = -0.3
            reward += y_reward

        # robot passed the stair
        passed_stair_reward = 0
        if self.back_leg_position > 0.540 and self.passed_stair_flag == False:
            self.passed_stair_flag = True
            passed_stair_reward = 200 / self.counts
            reward = passed_stair_reward
        # reach target
        reach_target_reward = 0
        if self.back_leg_position > self.target[0] and self.reach_target_flag == False:
            self.reach_target_flag = True
            reach_target_reward = 400 / self.counts
            reward = reach_target_reward
        falling_reward = 0
        if fall:
            falling_reward = -0.3
            reward += falling_reward

        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_10_05_try_2(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 2
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 200 / self.counts
        else:
            reach_target_reward = 0
        print("self.average_tips_x_position = {}".format(self.average_tips_x_position))
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > -0.01:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_10_05_try_3(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 2
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 1000 / self.counts
        else:
            reach_target_reward = 0
        print("self.average_tips_x_position = {}".format(self.average_tips_x_position))
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > - 0.015:
            not_moving_forward_reward = - 0.1
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        if self.counts > 50:
            reward = - 50
        return reward

    def reward_function_10_05_try_1(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 2
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 200 / self.counts
        else:
            reach_target_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / 1.2) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > -0.01:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_09_05_try_2(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 50
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 100 * 16 / self.counts
        else:
            reach_target_reward = 0
        x_reward = 5 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > 0:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    # החלוקה בx target לא טובה- האקספוננט גדל מהר מידי והרובוט מעדיף להתקדם בממוצע הרגליים במקום להתקדם לפרס "הגעה למטרה"
    def reward_function_08_05_try_1(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 1
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 2 * 16 / self.counts
        else:
            reach_target_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > 0:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_09_05_try_1(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 1
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 5 * 16 / self.counts
        else:
            reach_target_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / 1.5) - 5)
        print("self.average_tips_x_position")
        print(self.average_tips_x_position)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > 0:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    # החלוקה בx target לא טובה- האקספוננט גדל מהר מידי והרובוט מעדיף להתקדם בממוצע הרגליים במקום להתקדם לפרס "הגעה למטרה"
    def reward_function_08_05_try_1(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 1
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 2 * 16 / self.counts
        else:
            reach_target_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > 0:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    # reward שמצליח לעלות מדרגה
    def reward_function_04_05_try_1(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 1
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 2 * 16 / self.counts
        else:
            reach_target_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
        not_moving_forward_reward = 0
        # if the robot stay in place he doesn't get the x reward
        if average_tips_x_position_old - self.average_tips_x_position > 0:
            not_moving_forward_reward = - 0.02
            x_reward = 0
            passed_stair_reward = 0
            reach_target_reward = 0
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + passed_stair_reward + x_reward + y_reward + not_moving_forward_reward
        falling_reward = 0
        if fall:
            falling_reward = - 0.3
            reward = falling_reward
        print("reach_target_reward = {}, passed_stair_reward = {}, falling_reward = {}, x_reward = {}, y_reward = {}, "
              "not_moving_forward_reward = {}".format(reach_target_reward, passed_stair_reward, falling_reward,
                                                      x_reward, y_reward, not_moving_forward_reward))
        return reward

    def reward_function_03_05_try_1(self, leg, fall, reach_target, passed_stair, average_tips_x_position_old):
        if passed_stair:
            passed_stair_reward = 1
        else:
            passed_stair_reward = 0
        if reach_target:
            reach_target_reward = 2
        else:
            reach_target_reward = - 0.02
        if fall:
            falling_reward = - 0.3
        else:
            falling_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
        if np.abs((self.body_position[1] - self.target[1])) > 0.25:
            y_reward = - 0.3
        else:
            y_reward = 0
        if average_tips_x_position_old - self.average_tips_x_position < 0.02:
            not_moving_forward_reward = - 0.02
        else:
            not_moving_forward_reward = 0
        reward = reach_target_reward + passed_stair_reward + falling_reward + x_reward + y_reward + not_moving_forward_reward
        return reward

    def reward_function_01_05_try_1(self, leg, fall, reach_target):
        if reach_target:
            reach_target_reward = 2
        else:
            reach_target_reward = - 0.02
        if fall:
            falling_reward = - 0.3
        else:
            falling_reward = 0
        x_reward = 2 * np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
        if np.abs((self.body_position[1] - self.target[1])) > 0.1:
            y_reward = - 0.3
        else:
            y_reward = 0
        reward = reach_target_reward + falling_reward + x_reward + y_reward
        return reward

    # def reward_function_30_04_try_1(self, leg, done, leg_tip_positions_world_frame_old):
    #     falling_reward = - 0.3
    #     x_reward = 2 * np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
    #     if np.abs((self.body_position[1] - self.target[1])) > 0.1:
    #         y_reward = - 0.3
    #     else:
    #         y_reward = 0
    #     min_angle = calculate_angle(leg_tip_positions_world_frame_old, leg)
    #     angle_reward = (min_angle/60)/100
    #     reward = x_reward + y_reward + angle_reward
    #     if done:
    #         reward += falling_reward
    #     return reward

    # def reward_function_27_04_try_1(self, leg, done, leg_tip_positions_world_frame_old):
    #     falling_reward = - 0.01
    #     x_reward = np.exp(5 * (self.average_tips_x_position / self.target[0]) - 5)
    #     if np.abs((self.body_position[1] - self.target[1])) > 0.05:
    #         y_reward = - 0.01
    #     else:
    #         y_reward = 0
    #     min_angle = calculate_angle(leg_tip_positions_world_frame_old, leg)
    #     angle_reward = (min_angle/60)/100
    #     # polygon_size = calculate_projection_polygon_area(leg_tip_positions_body_frame_old, leg)
    #     # polygon_reward = - polygon_size
    #     reward = x_reward + y_reward + angle_reward
    #     if done:
    #         reward += falling_reward
    #     return reward
    #
    # def reward_function_25_04_try_1(self, leg, done, leg_tip_positions_body_frame_old):
    #     falling_reward = -0.01
    #     x_reward = np.exp(5 * (self.body_position[0] / self.target[0]) - 5) - 1
    #     y_reward = - np.abs((self.body_position[1] - self.target[1])) / 10
    #     # polygon_size = calculate_projection_polygon_area(leg_tip_positions_body_frame_old, leg)
    #     # polygon_reward = - polygon_size
    #     reward = x_reward + y_reward
    #     if done:
    #         reward += falling_reward
    #     return reward


def calculate_projection_polygon_area(points, I):
    # Get the three other points excluding the one at index I
    other_points = np.delete(points, I * 3 + np.arange(-3, 0))
    vertices = np.array(
        [[other_points[0], other_points[1]], [other_points[3], other_points[4]], [other_points[6], other_points[7]]])
    # Calculate the area of the triangle using the cross product method
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    area = 0.5 * np.abs(np.cross(v1, v2))

    return area


def calculate_angle(points, I):
    # Get the three other points excluding the one at index I
    other_points = np.delete(points, I * 3 + np.arange(-3, 0))
    x1, y1, x2, y2, x3, y3 = other_points[0], other_points[1], other_points[2], other_points[3], other_points[4], \
        other_points[5]
    # print("x1, y1, x2, y2, x3, y3 = ")
    # print(x1, y1, x2, y2, x3, y3)
    # Calculate the lengths of the sides of the triangle
    a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Calculate the angles using the Law of Cosines
    angle_a = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    angle_b = math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
    angle_c = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

    # Find the minimum angle
    min_angle = min(angle_a, angle_b, angle_c)

    # Convert the angle from radians to degrees
    min_angle_degrees = math.degrees(min_angle)

    return min_angle_degrees

# def reward_function_23_04_try_1(self):
#     distance_to_target_old = self.dis_to_target
#     # Compute the Euclidean distance between the target and current position
#     distance_to_target = ((self.body_position[0] - self.target[0]) ** 2 +
#                           (self.body_position[1] - self.target[1]) ** 2 +
#                           (self.body_position[2] - self.target[2]) ** 2) ** 0.5
#     print(distance_to_target)
#     # Define the rewards for reaching the target and moving closer to the target
#     target_reward = 100.0
#     distance_reward = 30 * np.exp(-(distance_to_target**3))
#     action_penalty = 3
#     # Compute the reward based on the distance to the target and the action taken
#     if distance_to_target < 0.1:
#         # If the agent reaches the target, give a large reward
#         reward = target_reward
#     else:
#         # Otherwise, give a reward based on the distance to the target and the action taken
#         reward = distance_reward
#
#     # Penalize the agent for taking an action that moves away from the target
#     if distance_to_target - distance_to_target_old > 0:
#         reward -= action_penalty
#
#     self.dis_to_target = distance_to_target
#     return reward

# def reward_function_20_04_try_2(self):
#     distance_to_target_old = self.dis_to_target
#     # Compute the Euclidean distance between the target and current position
#     distance_to_target = ((self.body_position[0] - self.target[0]) ** 2 +
#                           (self.body_position[1] - self.target[1]) ** 2 +
#                           (self.body_position[2] - self.target[2]) ** 2) ** 0.5
#     print(distance_to_target)
#     # Define the rewards for reaching the target and moving closer to the target
#     target_reward = 100.0
#     distance_reward = 30 * np.exp(-(distance_to_target**3))
#     action_penalty = -2
#     # Compute the reward based on the distance to the target and the action taken
#     if distance_to_target < 0.1:
#         # If the agent reaches the target, give a large reward
#         reward = target_reward
#     else:
#         # Otherwise, give a reward based on the distance to the target and the action taken
#         reward = distance_reward
#
#     # Penalize the agent for taking an action that moves away from the target
#     if distance_to_target - distance_to_target_old > 0:
#         reward -= action_penalty
#
#     self.dis_to_target = distance_to_target
#     return reward
#
# def reward_function_19_04_try_2(self):
#     distance_to_target_old = self.dis_to_target
#     # Compute the Euclidean distance between the target and current position
#     distance_to_target = ((self.body_position[0] - self.target[0]) ** 2 +
#                           (self.body_position[1] - self.target[1]) ** 2 +
#                           (self.body_position[2] - self.target[2]) ** 2) ** 0.5
#     print(distance_to_target)
#     # Define the rewards for reaching the target and moving closer to the target
#     target_reward = 150.0
#     distance_reward = 1 / distance_to_target
#
#     # Define the penalty for taking an action that moves away from the target
#     action_penalty = 0.1
#
#     # Compute the reward based on the distance to the target and the action taken
#     if distance_to_target < 0.2:
#         # If the agent reaches the target, give a large reward
#         reward = target_reward
#     else:
#         # Otherwise, give a reward based on the distance to the target and the action taken
#         reward = distance_reward
#
#         # Penalize the agent for taking an action that moves away from the target
#         if distance_to_target > distance_to_target_old:
#             reward -= action_penalty
#         else:
#             reward += action_penalty
#     self.dis_to_target = distance_to_target
#     return reward
#
# def reward_function_20_04_23_try_1(self):
#     distance_to_target_old = self.dis_to_target
#     # Compute the Euclidean distance between the target and current position
#     distance_to_target = ((self.body_position[0] - self.target[0]) ** 2 +
#                           (self.body_position[1] - self.target[1]) ** 2 +
#                           (self.body_position[2] - self.target[2]) ** 2) ** 0.5
#     print(distance_to_target)
#     # Define the rewards for reaching the target and moving closer to the target
#     target_reward = 100.0
#     distance_reward = 10 * np.exp(-(distance_to_target**3))
#
#     # Compute the reward based on the distance to the target and the action taken
#     if distance_to_target < 0.1:
#         # If the agent reaches the target, give a large reward
#         reward = target_reward
#
#     else:
#         # Otherwise, give a reward based on the distance to the target and the action taken
#         reward = distance_reward
#
#     self.dis_to_target = distance_to_target
#     return reward
