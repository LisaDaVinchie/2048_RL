import matplotlib.pyplot as plt
from utils.read_results import read_data
from pathlib import Path
import numpy as np
import json
import argparse
from utils.read_results import find_latest_file

# Parse the arguments
parser = argparse.ArgumentParser(description="Plot the results")
parser.add_argument('--paths', type = Path, required=True, help = "Path to the JSON file containing the paths.")
args = parser.parse_args()
paths_file_path = args.paths

# Load the paths
with open(paths_file_path) as f:
    paths = json.load(f)
    
final_result_basename = Path(paths["final_result_basename"])
result_file_ext = paths["result_file_ext"]
figure_basename = Path(paths["figure_basename"])
figure_ext = paths["figure_ext"]

IDX = None

if IDX is not None:
    file_path = f"{final_result_basename}_{IDX}{result_file_ext}"
else:
    file_path, IDX = find_latest_file(final_result_basename, result_file_ext)

print(f"Reading data from: {file_path}")
# final_scores, max_values, epsilons, loss, useless_moves = read_data(file_path)

with open(file_path, "r") as f:
    lines = f.readlines()

# Extract the values after each section header
final_scores = []
max_values = []
epsilon = []
loss = []
useless_moves = []
valid_moves = []

# Iterate through lines
for i, line in enumerate(lines):
    if line.startswith("Final score:"):
        final_scores = list(map(float, lines[i+1].split()))
    elif line.startswith("Max value reached:"):
        max_values = list(map(float, lines[i+1].split()))
    elif line.startswith("Epsilon:"):
        epsilon = list(map(float, lines[i+1].split()))
    elif line.startswith("Loss:"):
        loss = list(map(float, lines[i+1].split()))
    elif line.startswith("Useless moves:"):
        useless_moves = list(map(float, lines[i+1].split()))
    elif line.startswith("Valid moves:"):
        valid_moves = list(map(float, lines[i+1].split()))
# Plot the data
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

axs[0, 0].plot(final_scores)
axs[0, 0].set_title("Final Scores")
axs[0, 0].set_ylabel("Score")

axs[0, 1].plot(np.log2(max_values))
axs[0, 1].set_title("Max Value reached")
axs[0, 1].set_ylabel("Log2(Value)")
axs[0, 1].set_yticks(range(int(min(np.log2(max_values))), int(max(np.log2(max_values))) + 1))

axs[1, 0].plot(epsilon)
axs[1, 0].set_title("Epsilon")
axs[1, 0].set_xlabel("Episode")
axs[1, 0].set_ylabel("Value")

useless_moves_ratio = np.array(useless_moves) / (np.array(valid_moves) + np.array(useless_moves))

axs[1, 1].plot(useless_moves_ratio)
axs[1, 1].set_title("Useless Moves")
axs[1, 1].set_xlabel("Episode")
axs[1, 1].set_ylabel("Value")

plt.tight_layout()

# Save the figure
figure_path = f"{figure_basename}_{IDX}{figure_ext}"
plt.savefig(figure_path)