import numpy as np
import argparse
from pathlib import Path
import json
from time import time
from game_class import Game2048_env # Import the game class
start_time = time()

parser = argparse.ArgumentParser()
parser.add_argument('--params', type = Path, required=True, help = "Path to the JSON file containing the parameters.")
parser.add_argument('--paths', type = Path, required=True, help = "Path to the JSON file containing the paths.")
args = parser.parse_args()
paths_file_path = args.paths
params_file_path = args.params

with open(paths_file_path) as f:
    paths = json.load(f)
results_path = Path(paths["random_result_path"])

if not results_path.parent.exists():
    raise FileNotFoundError(f"The path {results_path} does not exist")

with open(params_file_path) as f:
    params = json.load(f)
    
grid_size = params["agent"]["grid_size"]
action_size = params["agent"]["action_size"]
n_channels = params["agent"]["n_channels"]

game_env = Game2048_env(size=grid_size)

max_values = []
final_score = []
actions = []

n_games = 100

i = 0
while i < n_games:
    state = game_env.reset()
    done = False
    i += 1
    max_value = 0
    old_reward = 0
    total_reward = 0
    
    game_actions = []
    while not done:
        action = np.random.randint(0, action_size)
        game_actions.append(action)
        
        next_state, done, merge_reward = game_env.step(state, action)
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
        
    max_values.append(max_value)
    final_score.append(total_reward)
    actions.append(game_actions)
    


elapsed_time = time() - start_time
with open(results_path, 'w') as f:
    f.write('Time taken [s]: ')
    f.write(str(elapsed_time))
    f.write('\n\n')
    
    f.write("Games played:\n")
    f.write(str(n_games))
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
