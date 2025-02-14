import numpy as np
import random

class Game2048_env:
    def __init__(self, size: int=4):
        "Initialize grid size and grid"
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        
    def reset(self):
        """Reset the grid and add a new tile"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self._add_new_tile()
        return self.grid
    
    def _add_new_tile(self, grid: np.ndarray) -> np.ndarray:
        """Add a new tile in the grid"""
        empty_cells = list(zip(*np.where(grid == 0)))
        
        # If there are empty cells in the grid, add a new tile in a random empty cell
        if empty_cells:
            row, col = random.choice(empty_cells)
            grid[row, col] = random.choice([2, 4])
        return grid
    
    def _merge_tiles(self, line: np.ndarray) -> np.ndarray:
        """Merge the tiles of a specific row or column"""
        non_zero_tiles = [tile for tile in line if tile != 0]
        merged_line = []
        skip = False
        n_tiles = len(non_zero_tiles)
        
        for i in range(n_tiles):
            if skip:
                skip = False
                continue
            if i + 1 < n_tiles and non_zero_tiles[i] == non_zero_tiles[i + 1]:
                merged_line.append(non_zero_tiles[i] * 2)
                skip = True
            else:
                merged_line.append(non_zero_tiles[i])
        
        merged_line += [0] * (len(line) - len(merged_line))
        return np.array(merged_line)

    def _move(self, grid: np.ndarray, direction: int):
        """Move the tiles of the grid in a specific direction"""
        # Up: transpose the grid
        if direction == 0:
            grid = grid.T
        # Down: transpose the grid and flip it
        elif direction == 1:
            grid = np.flipud(grid).T
        # Left: do nothing
        elif direction == 2:
            pass
        # Right: flip the grid
        elif direction == 3:
            grid = np.fliplr(grid)
        else:
            raise ValueError('Invalid direction')
        
        new_grid = np.array([self._merge_tiles(row) for row in grid])
        
        # Undo the transposition or the flip
        # Up: transpose the grid
        if direction == 0:
            new_grid = new_grid.T
        # Down: transpose the grid and flip it
        elif direction == 1:
            new_grid = np.flipud(new_grid.T)
        # Left: do nothing
        elif direction == 2:
            pass
        # Right: flip the grid
        elif direction == 3:
            new_grid = np.fliplr(new_grid)
        return new_grid

    def is_game_over(self, grid: np.ndarray) -> bool:
        """Check if the game is over"""
        if np.any(grid == 0):
            return False
        
        for row in range(self.size):
            for col in range(self.size):
                # Check if there are two adjacent tiles with the same value
                if row + 1 < self.size and grid[row, col] == grid[row + 1, col]:
                    return False
                if col + 1 < self.size and grid[row, col] == grid[row, col + 1]:
                    return False
                
        return True

    def step(self, grid: np.ndarray, direction: int) -> np.ndarray:
        """Play a step of the game"""
        new_grid = self._move(grid, direction)
        
        # If the grid has not changed, return the grid as it is
        if np.array_equal(grid, new_grid):
            return grid
        new_grid = self._add_new_tile(new_grid)
        return new_grid