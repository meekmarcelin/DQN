import sys
import os
from environment.custom_env import CustomCarEnv
from models.ddqn_model import build_model, build_agent

# Initialize the environment and load the trained model
env = CustomCarEnv()
model = build_model(input_shape=(96, 96, 3), action_space=env.action_space.n)
dqn = build_agent(model, env.action_space.n)
dqn.load_weights('models/ddqn_model.h5')

# Simulate using the trained model
dqn.test(env, nb_episodes=5, visualize=True)
env.close()
print("Simulation completed.")
