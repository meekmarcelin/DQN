# DQN-Based Car Driving Project

This project implements a Deep Q-Network (DQN) algorithm to train an agent to navigate and drive a car in a simulated environment. The agent learns to make decisions based on the visual inputs from the car's camera, with the goal of maximizing the reward (e.g., reaching a destination, avoiding obstacles, maintaining speed).


Video Demonstration


Link to video: https://www.loom.com/share/c791b4d39bb34b778232fb46188c7f59?sid=9fc265b9-2530-4080-a79f-0553d83012c2

## Project Structure

The project is organized into the following directories and files:

This project implements a Deep Q-Network (DQN) algorithm to train an agent to navigate and drive a car in a simulated environment. The agent learns to make decisions based on the visual inputs from the car's camera, with the goal of maximizing the reward (e.g., reaching a destination, avoiding obstacles, maintaining speed).

## Project Structure

The project is organized into the following directories and files:
DQN-Car/
├── environment/
│   ├── init.py
│   ├── car_env.py
├── models/
│   ├── init.py
│   ├── dqn_model.py
├── play.py
├── train.py
├── test_env.py
└── requirements.txt

- **environment**: Contains the definition of the custom car driving environment, including the car dynamics, observation space, and reward functions.
- **models**: Includes the code to build and train the DQN model for the car driving task.
- **play.py**: Runs the simulation using the trained DQN model to control the car.
- **train.py**: Trains the DQN model using the car driving environment.
- **test_env.py**: Tests the car driving environment setup and runs the environment with random actions.
- **requirements.txt**: Lists the required Python packages for the project.

## Getting Started

1. **Set up the virtual environment**:
   - Create a new virtual environment using `python -m venv env`.
   - Activate the virtual environment:
     - Windows: `env\Scripts\activate`
     - macOS/Linux: `source env/bin/activate`
2. **Install dependencies**:
   - Run `pip install -r requirements.txt` to install the required packages.
3. **Run the scripts**:
   - `python test_env.py`: Tests the car driving environment setup.
   - `python train.py`: Trains the DQN model for the car driving task.
   - `python play.py`: Runs the simulation with the trained DQN model controlling the car.

## Project Details

The `car_env.py` file defines the custom car driving environment, including the car's dynamics, sensor inputs (e.g., camera view, speed, steering angle), and reward functions. The agent must learn to navigate the car through the environment, avoiding obstacles and reaching the desired destination, by taking actions (e.g., accelerate, brake, steer).

The `dqn_model.py` file contains the code to build and train the DQN model for the car driving task. The model takes the car's sensor inputs as the observation space and outputs the Q-values for each possible action, which the agent uses to determine the best action to take.

The `train.py` script is responsible for training the DQN agent in the car driving environment, while the `play.py` script uses the trained model to control the car's movements in a simulated environment.


