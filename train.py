import gymnasium as gym
import numpy as np
from crop_disease_env import CropDiseaseEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Initialize the environment
env = CropDiseaseEnv()
np.random.seed(42)
env.seed(42)

# Define the neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# Define the policy, memory, and DQN agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save the trained model
dqn.save_weights('dqn_crop_disease_weights.h5f', overwrite=True)

# Test the trained model
dqn.test(env, nb_episodes=5, visualize=True)
