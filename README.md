# PACMAN_5200_DQN
Automation of PACMAN Version 5200 using reinforced learning. Version 1.

## Info
This repository contains a Deep Q-Network (DQN) implementation for training an agent to play Pac-Man using the Gym Retro environment. The model employs a Convolutional Neural Network (CNN) to process visual game states and outputs actions to maximize the agent's score.

A CNN processes image frames and outputs Q-values for each possible action.

The DQNAgent manages the training process, experience replay, action selection, and epsilon decay. It uses a deque for memory to store recent experiences and supports sampling.

The calculate_reward function rewards the agent for scoring and penalizes it for losing lives
## Results
After training for 50 epochs, with 5000 simulation steps per epoch, the best score achieved was 670 points. 
![](output.gif "Result.")
## To-do
Things that can be improved
* add a reward for time in simulation 
* add a penalty for standing or moving along routes without points 
* improve memory updating

## Installation guide

#### Install requirements
```
pip3 install -r requirements.txt
```

#### Put .nes file in Pacman_5200/ directory. Import the ROM using the import script
```
python3 -m retro.import Pacman_5200/
```

## Start training
```
python3 pacman_RL.py
```
## Use training weights
```
python3 test.py
```