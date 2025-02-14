import torch as th
import argparse
from pathlib import Path

from utils.import_params_json import load_config
from agent_class import DQN_Agent # Import the DQN_Agent class
from CNN_model import CNN_model # Import the neural network model class
from game_class import Game2048_env # Import the game class

# Load the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--params', type = Path, required=True, help = "Path to the JSON file containing the parameters.")
args = parser.parse_args()
params_file_path = args.params
config = load_config(params_file_path, ["agent", "CNN_model"])

# Initialise the model
grid_size: int = None
action_size: int = None
middle_channels: list = None
kernel_sizes: list = None
padding: list = None
locals().update(config["CNN_model"])  # Add variables to the local namespace
model = CNN_model(grid_size, action_size, middle_channels, kernel_sizes, padding)

# Choose loss and optimizer
loss_function = th.nn.MSELoss()
optimizer = th.optim.Adam

# Initialise the agent
n_episodes: int = None
state_size: int = None
action_size: int = None
gamma: float = None
epsilon: float = None
epsilon_decay: float = None
epsilon_min: float = None
buffer_maxlen: int = None
batch_size: int = None
locals().update(config["agent"])  # Add variables to the local namespace

agent = DQN_Agent(model, loss_function, optimizer,
                  state_size, action_size, gamma,
                  epsilon, epsilon_decay, epsilon_min,
                  buffer_maxlen, batch_size)

locals().update(config["agent"])

# Initialise the game environment
game_env = Game2048_env(grid_size)
final_scores = []

for episode in range(n_episodes):
    
    state = game_env.reset() # Put the grid in the initial state
    done = False # Initialize the done variable
    total_reward = 0 # Initialize the total reward
    is_action_exploratory = [] # Initialize the list of exploratory actions
    
    while not done:
        # Choose an action
        action, is_exploratory = agent.choose_action(state, training=True)
        
        # Take the action and observe the next state and reward
        next_state, reward, done = game_env.step(action)
        
        # Store the experience in the replay buffer
        agent.store_to_buffer(state, action, reward, next_state, done)
        
        # Update the state
        state = next_state
        total_reward += reward # Update the total reward
        is_action_exploratory.append(is_exploratory) # Store the exploratory action
    
    # Store the final score
    final_scores.append(total_reward)
        
    # Train the model
    agent.train_step()