from pathlib import Path

# Function to read the data from the file
def read_data(file_path: Path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract the values after each section header
    final_scores = []
    max_values = []
    epsilon = []
    loss = []

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

    return final_scores, max_values, epsilon, loss

# Find the path with the greatest index
def find_latest_file(basepath: Path, result_file_ext: str = ".txt") -> list:
    directory = basepath.parent
    basename = basepath.stem
    max_index = -1
    latest_file = None

    for file in directory.glob(f"{basename}_*{result_file_ext}"):
        try:
            index = int(file.stem.split('_')[-1])
            if index > max_index:
                max_index = index
                latest_file = file
        except ValueError:
            continue

    return latest_file, max_index