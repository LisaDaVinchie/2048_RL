import numpy as np
import random

class Game2048_env:
    def __init__(self, size: int=4):
        "Initialize grid size and grid"
        self.size = size
        
    def reset(self):
        """Reset the grid and add a new tile"""
        grid = np.zeros((self.size, self.size), dtype=int)
        self._add_new_tile(grid)
        return grid
    
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

    def _move(self, grid: np.ndarray, action: int) -> np.ndarray:
        """Move the tiles of the grid in a specific direction."""
        new_grid = np.zeros((self.size, self.size), dtype=int)
        
        if action == 0:  # Up
            for col in range(self.size):
                new_grid[:, col] = self._merge_tiles(grid[:, col])  # Merge column-wise
                
        elif action == 1:  # Down
            for col in range(self.size):
                new_grid[:, col] = np.flip(self._merge_tiles(np.flip(grid[:, col])))  # Reverse merge for downward motion
                
        elif action == 2:  # Left
            for row in range(self.size):
                new_grid[row, :] = self._merge_tiles(grid[row, :])  # Merge row-wise

        elif action == 3:  # Right
            for row in range(self.size):
                new_grid[row, :] = np.flip(self._merge_tiles(np.flip(grid[row, :])))  # Reverse merge for rightward motion
        else:
            raise ValueError("Invalid direction")
        
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

    def step(self, old_grid: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
        """Play a step of the game"""
        
        # Calculate the new grid after the move
        new_grid = self._move(old_grid, action)
        # Check if the game is over
        is_game_over = self.is_game_over(new_grid)
        
        # Check if the grid has changed
        if not np.array_equal(old_grid, new_grid):
            # If the grid has changed, add a new tile and check if the game is over
            new_grid = self._add_new_tile(new_grid)
        
        return new_grid, is_game_over