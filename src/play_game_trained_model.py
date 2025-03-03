import numpy as np
import torch as th
import argparse
from pathlib import Path
import json
from time import time
from game_class import Game2048_env # Import the game class
from agent_class import DQN_Agent
from models import LinearModel, CNN_model, Large_CNN
from utils.representations import to_one_hot, to_log2
start_time = time()

parser = argparse.ArgumentParser()
parser.add_argument('--params', type = Path, required=True, help = "Path to the JSON file containing the parameters.")
parser.add_argument('--paths', type = Path, required=True, help = "Path to the JSON file containing the paths.")
args = parser.parse_args()
paths_file_path = args.paths
params_file_path = args.params

with open(paths_file_path) as f:
    paths = json.load(f)
autoplay_path = Path(paths["autoplay_path"])
weights_folder = Path(paths["weights_folder"])
weights_basename = str(paths["weights_basename"])
weights_ext = str(paths["weights_ext"])

if not autoplay_path.parent.exists():
    raise FileNotFoundError(f"The path {autoplay_path.parent} does not exist")
if not weights_folder.exists():
    raise FileNotFoundError(f"The path {weights_folder} does not exist")
if weights_ext is None:
    raise ValueError("The weights extension is not defined")

with open(params_file_path) as f:
    params = json.load(f)
    
grid_size = params["agent"]["grid_size"]
action_size = params["agent"]["action_size"]
n_channels = params["agent"]["n_channels"]

n_games = params["test"]["n_epispdes"]
model_kind = params["test"]["model_kind"]
representation_kind = params["test"]["representation_kind"]
weight_number = params["test"]["weight_number"]

if representation_kind == "raw":
    representation = lambda x: x
elif representation_kind == "one_hot":
    representation = to_one_hot
elif representation_kind == "log2":
    representation = to_log2
else:
    raise ValueError("Invalid representation kind")

print(f"Playing {n_games} games with the model {model_kind} and the representation {representation_kind}", flush=True)

weights_path = Path(weights_basename + "_" + str(weight_number) + weights_ext)

game_env = Game2048_env(size=grid_size)
print(f"Game environment created with size {grid_size}\n", flush=True)

model = Large_CNN(params_path=params_file_path, middle_channels=[128, 128, 128, 128])
print(f"Model created\n", flush=True)

agent = DQN_Agent(params_path=params_file_path, model = model)
print(f"Agent created\n", flush=True)

agent.load(weights_path)
print(f"Weights loaded from {weights_path}\n", flush=True)

max_values = []
final_score = []
actions = []

i = 0
useless_move: int = 0
while i < n_games:
    print(f"Game {i}\n", flush=True)
    state = game_env.reset()
    next_state = state
    done = False
    i += 1
    max_value = 0
    old_reward = 0
    total_reward = 0
    
    game_actions = []
    j = 0
    while not done:
        j += 1
        model_state = representation(state, n_channels)
        # print(f"Step {j}", end="\r", flush=True)
        # print("------------------------------------")
        # print(state)
        # print("------------------------------------")
        
        k = 0
        while np.array_equal(state, next_state) and not done:
            action, is_exploratory = agent.choose_action(model_state, training=False)
            next_state, done, merge_reward = game_env.step(state, action)
            useless_move += 1
            k += 1
            if k > 10:
                action = np.random.randint(0, action_size)
                next_state, done, merge_reward = game_env.step(state, action)
        
        game_actions.append(action)
        reward = merge_reward - old_reward
        
        if not done and np.array_equal(state, next_state):
            reward -= 10
        total_reward += reward
        
        # Update the maximum value reached
        new_max = np.max(next_state)
        if new_max > max_value:
            max_value = new_max
        
        if done:
            break
        state = next_state
        old_reward = reward
    print("", flush=True)
        
    max_values.append(max_value)
    final_score.append(total_reward)
    actions.append(game_actions)
    


elapsed_time = time() - start_time
with open(autoplay_path, 'w') as f:
    f.write('Time taken [s]: ')
    f.write(str(elapsed_time))
    f.write('\n\n')
    
    f.write("Games played:\n")
    f.write(str(n_games))
    f.write("\n\n")
    
    f.write("Useless moves:\n")
    f.write(str(useless_move))
    f.write("\n\n")
    
    f.write("Final score:\n")
    for i, score in enumerate(final_score):
        f.write(f"{score}")
        if i < len(final_score) - 1:
            f.write("\t")
    f.write("\n\n")
    
    f.write("Max value reached:\n")
    for i, value in enumerate(max_values):
        f.write(f"{value}")
        if i < len(max_values) - 1:
            f.write("\t")
            
    f.write("\n\n")
    
    f.write("Actions:\n")
    for i, game_actions in enumerate(actions):
        for j, action in enumerate(game_actions):
            f.write(f"{action}")
            if j < len(game_actions) - 1:
                f.write("\t")
        f.write("\n")

