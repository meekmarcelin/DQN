import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__init__.py)))

from environment.custom_env import CustomCarEnv

env = CustomCarEnv()

for episode in range(5):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state

env.close()
print("Environment testing completed.")
