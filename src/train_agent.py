import torch as th
import numpy as np
import argparse
from pathlib import Path
import json
from time import time

from utils.import_params_json import load_config
from utils.visualize_game import print_grid, save_grid_to_file
from agent_class import DQN_Agent # Import the DQN_Agent class
from CNN_model import CNN_model # Import the neural network model class
from game_class import Game2048_env # Import the game class
from rewards import maxN_emptycells_reward

start_time = time()

# Load the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--params', type = Path, required=True, help = "Path to the JSON file containing the parameters.")
parser.add_argument('--paths', type = Path, required=True, help = "Path to the JSON file containing the paths.")
args = parser.parse_args()
paths_file_path = args.paths

with open(paths_file_path) as f:
    paths = json.load(f)
final_score_path = paths["final_result_path"]

params_file_path = args.params
config = load_config(params_file_path, ["agent", "CNN_model", "training"])

# Initialise the model
grid_size: int = None
action_size: int = None
middle_channels: list = None
kernel_sizes: list = None
padding: list = None
softmax: bool = None
locals().update(config["CNN_model"])  # Add variables to the local namespace
print("Using softmax = ", softmax)
model = CNN_model(grid_size, action_size, middle_channels, kernel_sizes, padding, softmax)
print("Model initialised\n")

learning_rate: float = None
n_episodes: int = None
print_every: int = None
locals().update(config["training"])

# Choose loss and optimizer
loss_function = th.nn.SmoothL1Loss()
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
target_update_freq: int = None
locals().update(config["agent"])  # Add variables to the local namespace

agent = DQN_Agent(model, loss_function, optimizer,
                  state_size, action_size, gamma,
                  epsilon, epsilon_decay, epsilon_min,
                  buffer_maxlen, batch_size, target_update_freq)

locals().update(config["agent"])
print("Agent initialised\n")

# Initialise the game environment
game_env = Game2048_env(size=grid_size)
print("Game environment initialised\n")

final_scores = []
max_value_reached = []
train_epsilon = []
train_loss = []
train_Q_values = []
reward_function = maxN_emptycells_reward

print("Training the agent\n")
for episode in range(n_episodes):
    if (episode + 1) % print_every == 0:
        print(f"Episode {episode + 1}/{n_episodes}")
        
    state = game_env.reset() # Put the grid in the initial state
    state = th.tensor(state, dtype=th.float32) # Convert the state to a tensor
    done = False # Initialize the done variable
    total_reward = 0 # Initialize the total reward
    is_action_exploratory = [] # Initialize the list of exploratory actions
    
    while not done:
        # Choose an action
        action, is_exploratory = agent.choose_action(state.unsqueeze(0).unsqueeze(0), training=True)
        
        # Take the action and observe the next state and reward
        next_state, done = game_env.step(state.numpy(), action)
        
        reward = reward_function(state.numpy(), next_state, done, params_file_path)
        
        next_state = th.tensor(next_state, dtype=th.float32) # Convert the next state to a tensor
        # Store the experience in the replay buffer
        agent.store_to_buffer(state.unsqueeze(0), action, reward, next_state.unsqueeze(0), done)
        
        total_reward += reward # Update the total reward
        
        state = next_state # Update the state
        is_action_exploratory.append(is_exploratory) # Store the exploratory action
    
    # Store the final score
    final_scores.append(total_reward)
    max_value_reached.append(np.max(state.numpy()))
        
    # Train the model
    agent.train_step(episode)
    
    train_epsilon.append(agent.epsilon)
    train_loss.append(agent.loss)
    train_Q_values.append(agent.current_q_values)
    if (episode + 1) % print_every == 0:
        print(f"Total reward: {total_reward}\n")
    

with open(params_file_path, 'r') as f:
    params = json.load(f)

json_str = json.dumps(params, indent=4)[1: -1]

i = 0
    
# Save the final scores to a file
with open(final_score_path, 'w') as f:
    f.write("Final score:\n")
    for i, score in enumerate(final_scores):
        f.write(f"{score}")
        if i < len(final_scores) - 1:
            f.write("\t")
    f.write("\n\n")
    f.write("Max value reached:\n")
    for i, value in enumerate(max_value_reached):
        f.write(f"{value}")
        if i < len(max_value_reached) - 1:
            f.write("\t")
    f.write("\n\n")
    f.write("Epsilon:\n")
    for i, eps in enumerate(train_epsilon):
        f.write(f"{eps}")
        if i < len(train_epsilon) - 1:
            f.write("\t")
    f.write("\n\n")
    f.write("Loss:\n")
    for i, loss in enumerate(train_loss):
        f.write(f"{loss}")
        if i < len(train_loss) - 1:
            f.write("\t")
    f.write("\n\n")
    f.write("Q values:\n")
    for i, q_values in enumerate(train_Q_values):
        f.write(f"{q_values}")
        if i < len(train_Q_values) - 1:
            f.write("\t")
    f.write("\n\n")
    
    f.write("Parameters:")
    f.write(json_str)
    
print(f"Final scores saved to {final_score_path}\n")

end_time = time()

print(f"Training completed in {((end_time - start_time) / 60):.4f} minutes")
        
