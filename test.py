import retro
import cv2
import torch
import numpy as np
from CNN_model import CNN
from video_writer import AsyncVideoWriter

weights_path = "/root/workspace/PACMAN_5200_DQN/best.pt"


class DQNAgent:
    def __init__(self, model, action_size=9):
        self.action_size = action_size
        self.model = model

    def act(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().cuda()
            act_values = self.model(state_tensor).cpu().numpy()
        return np.argmax(act_values[0])


model = CNN().cuda()
model.load_state_dict(torch.load(weights_path))

agent = DQNAgent(model=model)

# Initialize enviroment
env = retro.make(game="PacManNamco-Nes",
                 use_restricted_actions=retro.Actions.DISCRETE)
state = env.reset()
state = np.reshape(state, (1, 3, 240, 224)) / 255

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = "/root/workspace/PACMAN_5200_DQN/simulation_5_7.avi"
vw = AsyncVideoWriter(video_path, fourcc, 30, (240, 224))

done = False
total_reward = 0

while not done:
    # Get an action from agent
    action = agent.act(state)

    # Make a step
    next_state, reward, done, info = env.step(action)
    total_reward += reward

    frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
    vw.write(frame)

    # Prepare state for the next step
    state = np.reshape(next_state, (1, 3, 240, 224)) / 255

    if done:
        print(f"Total reward: {total_reward}")

vw.release()
env.close()
