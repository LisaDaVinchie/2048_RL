import torch as th
import numpy as np
import argparse
from pathlib import Path
import json
from time import time

from utils.import_params_json import load_config
# from utils.visualize_game import print_grid # Optionally print the grid
from utils.representations import to_one_hot, from_one_hot, to_log2, from_log2
from agent_class import DQN_Agent # Import the DQN_Agent class
from models import CNN_model, LinearModel, Large_CNN # Import the neural network model class
from game_class import Game2048_env # Import the game class
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
weights_file_path = paths["weights_file_path"]

params_file_path = args.params
config = load_config(params_file_path, ["training"])

learning_rate: float = None
n_episodes: int = None
print_every: int = None
model_kind: str = None
loss_kind: str = None
optimizer_kind: str = None
representation_kind: str = None
locals().update(config["training"])

config = load_config(params_file_path, ["rewards"])
no_changes_penalty = config["rewards"]["no_changes_penalty"]

print("No changes penalty: ", no_changes_penalty)


# Initialise the model
if model_kind == "linear":
    model = LinearModel(params_file_path)
elif model_kind == "cnn":
    model = CNN_model(params_file_path)
elif model_kind == "large_cnn":
    model = Large_CNN(params_file_path)
else:
    raise ValueError("Invalid model kind")

# Choose loss and optimizer
if loss_kind == "mse":
    loss_function = th.nn.MSELoss()
elif loss_kind == "huber":
    loss_function = th.nn.SmoothL1Loss()
else:
    raise ValueError("Invalid loss kind")

if optimizer_kind == "adam":
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_kind == "sgd":
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_kind == "rmsprop":
    optimizer = th.optim.RMSprop(model.parameters(), lr=learning_rate)
else:
    raise ValueError("Invalid optimizer kind")

# Initialise the agent

agent = DQN_Agent(params_file_path, model, loss_function, optimizer)
n_channels = agent.n_channels
# Initialise the game environment
game_env = Game2048_env(size=agent.grid_size)
print("Game environment initialised\n", flush=True)

if representation_kind == "raw":
    encode_function = lambda x: x
    decode_function = lambda x: x
elif representation_kind == "log2":
    encode_function = lambda x: to_log2(x, n_channels)
    decode_function = lambda x: from_log2(x, n_channels)
elif representation_kind == "one_hot":
    encode_function = lambda x: to_one_hot(x, n_channels)
    decode_function = lambda x: from_one_hot(x)
else:
    raise ValueError("Invalid representation kind")

final_scores = []
max_value_reached = []
train_epsilon = []
train_loss = []
train_Q_values = []
non_valid_moves = []
valid_moves = []

print("Training the agent\n", flush=True)
for episode in range(n_episodes):
    if (episode + 1) % print_every == 0:
        print(f"Episode {episode + 1}/{n_episodes}", flush=True)
        
    state = game_env.reset() # Put the grid in the initial state
    done = False # Initialize the done variable
    is_action_exploratory = [] # Initialize the list of exploratory actions
    
    agent.loss = None
    agent.current_state_values = None
    
    max_value = 0
    total_reward = 0 # Initialize the total reward
    old_reward = 0
    
    old_avg = np.mean(state)
    new_avg = 0
    
    n_non_valid_moves: int = 0
    n_valid_moves: int = 0
    while not done:
        # Choose an action
        stored_state = encode_function(state)
        
        action, is_exploratory = agent.choose_action(stored_state, training=True)
        
        # Take the action and observe the next state and reward
        next_state, done, merge_reward = game_env.step(state, action)
        reward = merge_reward
        old_reward = merge_reward
        
        if done:
            next_state = None
        
        if np.array_equal(next_state, state) and next_state is not None:
            reward += -10
            n_non_valid_moves += 1
        else:
            n_valid_moves += 1
        
        is_action_exploratory.append(is_exploratory) # Store the exploratory action
        
        
        # Update the maximum value reached
        new_max = np.max(next_state)
        if next_state is not None and new_max > max_value:
            max_value = new_max
        # if new_avg > old_avg:
        #     old_avg = new_avg
        #     reward += np.log2(new_avg)
        
        if done:
            break
        
        stored_next_state = encode_function(next_state)
        agent.store_to_buffer(stored_state, action, reward, stored_next_state, done)
        
        
        total_reward += reward # Update the total reward
        state = next_state # Update the state
        
    
    # Store the final score
    final_scores.append(total_reward)
    max_value_reached.append(max_value)
    non_valid_moves.append(n_non_valid_moves)
    valid_moves.append(n_valid_moves)
    
    # Train the model
    agent.train_step(episode)
    
    train_epsilon.append(agent.epsilon)
    train_loss.append(agent.loss)
    train_Q_values.append(agent.current_state_values)
    if (episode + 1) % print_every == 0:
        print(f"Average total reward: {np.sum(final_scores[-print_every:]) / print_every}", flush=True)
        print(f"Average max value reached: {np.sum(max_value_reached[-print_every:]) / print_every}\n", flush=True)
        
elapsed_time = time() - start_time

# Save the parameters to a file
with open(params_file_path, 'r') as f:
    params = json.load(f)

json_str = json.dumps(params, indent=4)[1: -1]

i = 0
    
# Save the final scores to a file
with open(final_score_path, 'w') as f:
    f.write('Time taken [s]: ')
    f.write(str(elapsed_time))
    f.write('\n\n')
    
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
    
    f.write("Useless moves:\n")
    for i, moves in enumerate(non_valid_moves):
        f.write(f"{moves}")
        if i < len(non_valid_moves) - 1:
            f.write("\t")
    f.write("\n\n")
    
    f.write("Valid moves:\n")
    for i, moves in enumerate(valid_moves):
        f.write(f"{moves}")
        if i < len(valid_moves) - 1:
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
    
    f.write("Parameters:")
    f.write(json_str)
    
agent.save(weights_file_path)
print("Model saved\n", flush=True) 
    
print(f"Final scores saved to {final_score_path}\n", flush=True)

print(f"Training completed in {((elapsed_time) / 60):.4f} minutes\n\n", flush=True)
        
