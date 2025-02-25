import torch as th
import numpy as np
import argparse
from pathlib import Path
import json
from time import time

from utils.import_params_json import load_config
from agent_class import DQN_Agent # Import the DQN_Agent class
from models import CNN_model, LinearModel, Large_CNN # Import the neural network model class
from game_class import Game2048_env # Import the game class
from rewards import maxN_emptycells_reward, original_reward, maxN_emptycells_merge_reward, log2_merge_reward, sum_maxval_reward
reward_function = original_reward
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
config = load_config(params_file_path, ["training"])

learning_rate: float = None
n_episodes: int = None
print_every: int = None
model_kind: str = None
loss_kind: str = None
optimizer_kind: str = None
locals().update(config["training"])

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

grid_size = agent.grid_size
# Initialise the game environment
game_env = Game2048_env(size=grid_size)
print("Game environment initialised\n", flush=True)

print("Heat up the replay buffer\n", flush=True)
# Heat up the replay buffer
while len(agent.replay_buffer) < agent.batch_size:
    state = game_env.reset()
    done = False
    while not done:
        action = np.random.randint(0, 4)
        old_reward = game_env.merge_reward
        next_state, done, merge_reward = game_env.step(state, action)
        # reward = reward_function(state.numpy(), next_state, done, params_file_path)
        reward = merge_reward - old_reward
        agent.store_to_buffer(state, action, reward, next_state, done)
        state = next_state

final_scores = []
max_value_reached = []
train_epsilon = []
train_loss = []
train_Q_values = []

print("Training the agent\n", flush=True)
for episode in range(n_episodes):
    if (episode + 1) % print_every == 0:
        print(f"Episode {episode + 1}/{n_episodes}", flush=True)
        
    state = game_env.reset() # Put the grid in the initial state
    done = False # Initialize the done variable
    is_action_exploratory = [] # Initialize the list of exploratory actions
    
    agent.loss = None
    agent.current_q_values = None
    
    max_value = 0
    total_reward = 0 # Initialize the total reward
    while not done:
        # Choose an action
        action, is_exploratory = agent.choose_action(state, training=True)
        old_reward = game_env.merge_reward
        
        # Take the action and observe the next state and reward
        next_state, done, merge_reward = game_env.step(state, action)
        
        total_reward += merge_reward - old_reward
        
        if not done and np.array_equal(state, next_state):
            total_reward -= 10
        
        agent.store_to_buffer(state, action, reward, next_state, done)
        
        state = next_state # Update the state
        is_action_exploratory.append(is_exploratory) # Store the exploratory action
        
        # Update the maximum value reached
        max_value = np.maximum(max_value, np.max(state))
    
    # Store the final score
    final_scores.append(total_reward)
    max_value_reached.append(max_value)
    
    # Train the model
    agent.train_step(episode)
    
    train_epsilon.append(agent.epsilon)
    train_loss.append(agent.loss)
    train_Q_values.append(agent.current_q_values)
    if (episode + 1) % print_every == 0:
        print(f"Total reward: {total_reward}\n", flush=True)
    
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
    
print(f"Final scores saved to {final_score_path}\n", flush=True)

print(f"Training completed in {((elapsed_time) / 60):.4f} minutes\n\n", flush=True)
        
