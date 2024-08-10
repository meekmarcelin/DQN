from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def build_model(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4, input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    return model

def build_agent(model, action_space):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=action_space, nb_steps_warmup=10, target_model_update=1e-2)
    dqn.compile('adam', metrics=['mae'])
    return dqn
