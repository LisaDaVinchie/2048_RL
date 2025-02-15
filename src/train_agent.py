import torch as th
import argparse
from pathlib import Path
import json

from utils.import_params_json import load_config
from agent_class import DQN_Agent # Import the DQN_Agent class
from CNN_model import CNN_model # Import the neural network model class
from game_class import Game2048_env # Import the game class
from rewards import maxN_emptycells_reward

# Load the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--params', type = Path, required=True, help = "Path to the JSON file containing the parameters.")
parser.add_argument('--paths', type = Path, required=True, help = "Path to the JSON file containing the paths.")
args = parser.parse_args()
paths_file_path = args.paths

with open(paths_file_path) as f:
    paths = json.load(f)
final_score_path = paths["final_score_path"]

params_file_path = args.params
config = load_config(params_file_path, ["agent", "CNN_model", "training"])

# Initialise the model
grid_size: int = None
action_size: int = None
middle_channels: list = None
kernel_sizes: list = None
padding: list = None
locals().update(config["CNN_model"])  # Add variables to the local namespace
model = CNN_model(grid_size, action_size, middle_channels, kernel_sizes, padding)
print("Model initialised\n")

learning_rate: float = None
n_episodes: int = None
print_every: int = None
locals().update(config["training"])

# Choose loss and optimizer
loss_function = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

# Initialise the agent
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
print("Agent initialised\n")

# Initialise the game environment
game_env = Game2048_env(params_file_path, maxN_emptycells_reward, grid_size)
print("Game environment initialised\n")

final_scores = []

print("Training the agent\n")
for episode in range(n_episodes):
    print(f"Episode {episode + 1}/{n_episodes}")
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
    print(f"Total reward: {total_reward}\n")
    
# Save the final scores to a file
with open('final_scores.txt', 'w') as f:
    for score in final_scores:
        f.write(f"{score}\n")
