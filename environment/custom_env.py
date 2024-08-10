import numpy as np
import pygame
import gym
from gym import spaces

class CustomCarEnv(gym.Env):
    def __init__(self):
        super(CustomCarEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # Actions: 0-Left, 1-Right, 2-Accelerate, 3-Brake, 4-No-op
        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)

        # Initialize the state
        self.state = np.zeros((96, 96, 3), dtype=np.uint8)  # Example state
        self.done = False

        # Initialize PyGame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Car Racing Environment")
        self.clock = pygame.time.Clock()

        # Load images from the assets folder
        self.car_image = pygame.image.load('assets/car.png').convert_alpha()  # Load car image with transparency
        self.track_image = pygame.image.load('assets/track.png').convert()    # Load track image

        # Set initial position of the car
        self.car_position = [150, 150]  # Adjust based on your track design

    def reset(self):
        # Reset the environment to an initial state and return the initial observation
        self.car_position = [150, 150]  # Reset car to starting position
        self.done = False
        return self.state

    def step(self, action):
        # Logic to update the state based on the action taken
        if action == 0:  # Move left
            self.car_position[0] -= 10
        elif action == 1:  # Move right
            self.car_position[0] += 10
        elif action == 2:  # Accelerate (move up)
            self.car_position[1] -= 10
        elif action == 3:  # Brake (move down)
            self.car_position[1] += 10

        # Check if the car goes out of bounds or reaches a goal (customize as needed)
        self.done = self.check_termination_conditions()

        # Return observation, reward, done, and info
        reward = 1.0  # Example reward
        return self.state, reward, self.done, {}

    def check_termination_conditions(self):
        # Define termination conditions, e.g., collision, goal reached, or time limit exceeded
        # Placeholder for actual termination logic
        return False

    def render(self, mode='human'):
        # Render the environment to the screen
        self.screen.blit(self.track_image, (0, 0))  # Draw the track as the background
        self.screen.blit(self.car_image, self.car_position)  # Draw the car at its current position
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
