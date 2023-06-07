# Quadruped-robot
Motion Planning for Robotic Dog Walking in Unstructured Terrain- Master thesis by Shirelle Marcus Drori.

The research video, containing the key highlights and main points, can be accessed through the following link: https://youtu.be/YQYZGnZ85pM
This video provides a concise overview of the research findings, allowing viewers to gain an understanding of the core aspects and significant contributions of the study.

The repository consists of three key components:

1. The main_code.ttt file, which serves as the CoppeliaSim scene file. This file encompasses the implementation of the stable step algorithm.

2. The RobotModelEnv_spot.py, which is a gym environment specifically designed for the training process. This environment provides the necessary framework for training and evaluating the performance of the robot.

3. The robot_dog_learning_spot.py script, which instantiates a Soft Actor-Critic (SAC) model and initiates the learning process. This script orchestrates the training procedure for the robot, enabling it to acquire the desired skills through the SAC algorithm.

This repository provides a comprehensive and structured framework for leveraging CoppeliaSim, the RobotModelEnv_spot gym environment, and the robot_dog_learning_spot.py script to train the robot dog using our algorithm.
