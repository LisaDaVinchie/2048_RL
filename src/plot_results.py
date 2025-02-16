import matplotlib.pyplot as plt
from utils.read_results import read_data
from pathlib import Path
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
final_scores, max_values = read_data(file_path)

print(f"Final scores shape: {len(final_scores)}")
print(f"Max values shape: {len(max_values)}")

# Plot the data
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(final_scores)
axs[0].set_title("Final Scores")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Score")

axs[1].plot(max_values)
axs[1].set_title("Max Value reached")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Value")

plt.tight_layout()

# Save the figure
figure_path = f"{figure_basename}_{IDX}{figure_ext}"
plt.savefig(figure_path)