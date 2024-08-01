import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CropDiseaseEnv(gym.Env):
    def __init__(self):
        super(CropDiseaseEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.agent_position = np.array([0, 0])
        self.cure_position = np.array([4, 4])
        self.disease_position = np.array([2, 2])

    def reset(self):
        self.agent_position = np.array([0, 0])
        return self.agent_position

    def step(self, action):
        if action == 0:  # Up
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # Down
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)
        elif action == 2:  # Left
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # Right
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)

        if np.array_equal(self.agent_position, self.cure_position):
            reward = 10
            done = True
        elif np.array_equal(self.agent_position, self.disease_position):
            reward = -10
            done = True
        else:
            reward = -1
            done = False

        return self.agent_position, reward, done, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[tuple(self.agent_position)] = 1
        grid[tuple(self.cure_position)] = 2
        grid[tuple(self.disease_position)] = 3
        print(grid)
