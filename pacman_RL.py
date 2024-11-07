import retro
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import os
from video_writer import AsyncVideoWriter
from CNN_model import CNN

reward_number = 0

results_path = '/root/workspace/PACMAN_5200_DQN/results_v3/'

if not os.path.exists(results_path):
    os.makedirs(results_path)
    print(f"The folder '{results_path}' was created.")
else:
    print(f"The folder '{results_path}' exists.")

model = CNN()
criterion = nn.SmoothL1Loss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


class DQNAgent:
    def __init__(self, action_size=9):
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward)
        self.memory.append((state, action, reward, next_state, done, priority))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().cuda()
            act_values = self.model(state_tensor).cpu().numpy()
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.choices(self.memory, weights=[
                                   i[5] for i in self.memory], k=batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done, _ in minibatch:
            state_tensor = torch.from_numpy(state).float().cuda()
            target = reward
            with torch.no_grad():
                if not done:
                    next_state_tensor = torch.from_numpy(
                        next_state).float().cuda()
                    target += self.gamma * \
                        torch.max(self.model(next_state_tensor)).item()
            target_f = self.model(state_tensor).cpu().detach().numpy()
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        self.train(torch.from_numpy(np.vstack(states)).float().cuda(),
                   torch.from_numpy(np.vstack(targets)).float().cuda())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, inputs, targets):
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def calculate_reward(info, previous_info, ghost_eat_multiplier):
    reward = 0
    score_increase = info['score'] - previous_info['score']

    # reward for gettinf scores
    if score_increase == 1:
        reward += 20
    elif score_increase == 5:
        reward += 100
        ghost_eat_multiplier = 20
    elif score_increase == ghost_eat_multiplier:
        reward += ghost_eat_multiplier*20
        ghost_eat_multiplier *= 2

    # penalty for loosing live
    if info['lives'] < previous_info['lives']:
        reward -= 50

    return reward, ghost_eat_multiplier


# Initialize environment
env = retro.make(game="PacManNamco-Nes",
                 use_restricted_actions=retro.Actions.DISCRETE)
state_size = env.observation_space.shape
action_size = env.action_space.n

agent = DQNAgent()
batch_size = 64
EPISODES = 10
max_steps = 5000

previous_lives = 3
previous_score = 0

# Initialize lists for ploting graphs
episode_rewards = []
episode_steps = []
epsilon_values = []

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, (1, 3, 240, 224)) / 255
    done = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = results_path + \
        f'Reward_{reward_number}_{e}_{datetime.datetime.now()}.avi'
    vw = AsyncVideoWriter(video_path, fourcc, 30, (240, 224))

    # Total reward per episode
    total_reward = 0
    previous_info = {'lives': 3, 'score': 0}
    ghost_eat_multiplier = 20

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        reward, ghost_eat_multiplier = calculate_reward(
            info, previous_info, ghost_eat_multiplier)
        total_reward += reward

        previous_info = info.copy()

        # Write frames for video
        vw.write(next_state)

        # Preparing for the next state and updating agent's memory
        next_state = np.reshape(next_state, (1, 3, 240, 224)) / 255
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        print(
            f"Step: {step}, Step reward: {reward}, Total reward: {total_reward}, Memory Size: {len(agent.memory)}")

        if done:
            vw.release()
            reward_number = total_reward
            agent.save(
                results_path + f'Reward_{reward_number}_Frames_{step}_Episode_{e}_{datetime.datetime.now()}.pt')
            print(
                f"Episode {e+1}/{EPISODES}, Steps Taken: {step}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Memory Size: {len(agent.memory)}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # Updating metrics for further plotting
        episode_rewards.append(total_reward)
        episode_steps.append(step)
        epsilon_values.append(agent.epsilon)

plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.savefig("reward.png")

plt.figure(figsize=(12, 6))
plt.plot(episode_steps, label="Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Number of Steps per Episode")
plt.legend()
plt.savefig("steps.png")

plt.figure(figsize=(12, 6))
plt.plot(epsilon_values, label="Epsilon Value")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay over Episodes")
plt.legend()
plt.savefig("epsilon.png")
