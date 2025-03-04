import torch as th
import numpy as np
import argparse
from pathlib import Path
import json
from time import time

from utils.import_params_json import load_config
from utils.visualize_game import print_grid
from utils.representations import to_one_hot, from_one_hot, to_log2, from_log2
from agent_class import DQN_Agent # Import the DQN_Agent class
from models import CNN_model, LinearModel, Large_CNN # Import the neural network model class
from game_class import Game2048_env # Import the game class
# from rewards import maxN_emptycells_reward, original_reward, maxN_emptycells_merge_reward, log2_merge_reward, sum_maxval_reward
# reward_function = original_reward
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
game_env = Game2048_env(size=agent.grid_size, n_channels=n_channels)
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

print("Heat up the replay buffer\n", flush=True)
# Heat up the replay buffer
i = 0
while len(agent.replay_buffer) < agent.batch_size:
    state = game_env.reset()
    done = False
    i += 1
    max_value = 0
    old_reward = 0
    while not done:
        action = np.random.randint(0, 4)
        next_state, done, merge_reward = game_env.step(state, action)
        reward = merge_reward - old_reward
        
        if not done and np.array_equal(state, next_state):
            reward += -no_changes_penalty
            
        # Update the maximum value reached
        new_max = np.max(next_state)
        if new_max > max_value:
            max_value = new_max
        
        stored_state = encode_function(state)
        stored_next_state = encode_function(next_state)
        agent.store_to_buffer(stored_state, action, reward, stored_next_state, done)
        if done:
            break
        state = next_state
        old_reward = reward

final_scores = []
max_value_reached = []
train_epsilon = []
train_loss = []
train_Q_values = []
useless_moves_perc = []

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
    
    n_no_move: int = 0
    n_moves: int = 0
    while not done:
        # Choose an action
        stored_state = encode_function(state)
        
        action, is_exploratory = agent.choose_action(stored_state, training=True)
        n_moves += 1
        # Take the action and observe the next state and reward
        next_state, done, merge_reward = game_env.step(state, action)
        reward = merge_reward - old_reward
        
        if np.array_equal(next_state, state) and not done:
            reward += -no_changes_penalty
            n_no_move += 1
        
        is_action_exploratory.append(is_exploratory) # Store the exploratory action
        
        
        # Update the maximum value reached
        new_max = np.max(next_state)
        if new_max > max_value:
            max_value = new_max
        
        if done:
            break
        
        old_reward = reward
        stored_next_state = encode_function(next_state)
        agent.store_to_buffer(stored_state, action, reward, stored_next_state, done)
        
        
        total_reward += reward # Update the total reward
        state = next_state # Update the state
    
    # Store the final score
    final_scores.append(total_reward)
    max_value_reached.append(max_value)
    useless_moves_perc.append(n_no_move / n_moves)
    
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
    for i, moves in enumerate(useless_moves_perc):
        f.write(f"{moves}")
        if i < len(useless_moves_perc) - 1:
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
        
