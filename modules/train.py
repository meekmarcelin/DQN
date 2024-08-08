import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
cd 