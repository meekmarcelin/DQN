import sys
import os
from environment.custom_env import CustomCarEnv
from models.ddqn_model import build_model, build_agent

# Initialize the environment and agent
env = CustomCarEnv()
model = build_model(input_shape=(96, 96, 3), action_space=env.action_space.n)
dqn = build_agent(model, env.action_space.n)

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Save the trained model weights
dqn.save_weights('models/ddqn_model.h5', overwrite=True)
print("Training completed and model saved.")
