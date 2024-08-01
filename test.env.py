import gymnasium as gym
import numpy as np
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.core import Processor
from crop_disease_env import CropDiseaseEnv

# Define a custom processor for normalizing/denormalizing state observations
class CustomProcessor(Processor):
    def process_observation(self, observation):
        return observation / 4.0  # Normalizes the observation (since grid size is 5x5)

    def process_state_batch(self, batch):
        return batch

    def process_reward(self, reward):
        return reward

# Load the environment and the model
env = CropDiseaseEnv()
model = load_model('dqn_crop_disease.h5')

# Create the DQN agent
dqn = DQNAgent(model=model, policy=GreedyQPolicy(), processor=CustomProcessor(),myenv\Scripts\activate


               nb_actions=env.action_space.n, memory=None, nb_steps_warmup=0)

# Set the processor to None since it is already included in the DQN agent
dqn.processor = None

# Simulate an episode using the trained model
state = env.reset()
done = False

while not done:
    env.render()  # Render the environment
    action = np.argmax(dqn.compute_q_values(state.reshape(1, -1)))  # Select the best action based on Q-values
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.render()  # Render the final state
print("Simulation ended.")