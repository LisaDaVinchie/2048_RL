
def print_grid(grid):
    print("\n")
    print("------------------------------------")
    for row in grid:
        print("\t".join(map(str, row)))
    print("------------------------------------")
    print("\n")

def save_grid_to_file(grid, file_path):
    with open(file_path, 'a') as f:
        f.write("\n")
        f.write("------------------------------------")
        for row in grid:
            f.write("\n")
            f.write(" ".join(map(str, row)))
        
        f.write("\n------------------------------------")
        f.write("\n")