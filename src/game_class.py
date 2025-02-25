import numpy as np
import random

class Game2048_env:
    def __init__(self, size: int=4, n_channels: int = 12):
        "Initialize grid size and grid"
        self.size = size
        self.max_number = 2 ** (n_channels - 1)
        
    def reset(self) -> np.ndarray:
        """Reset the grid and add a new tile"""
        grid = np.zeros((self.size, self.size), dtype=int)
        self._add_new_tile(grid)
        self.step(grid, 0)
        return grid
    
    def _add_new_tile(self, grid: np.ndarray) -> np.ndarray:
        """Add a new tile in the grid"""
        empty_cells = list(zip(*np.where(grid == 0)))
        
        # If there are empty cells in the grid, add a new tile in a random empty cell
        if empty_cells:
            row, col = random.choice(empty_cells)
            grid[row, col] = 2 if random.random() < 0.9 else 4
        return grid
    
    def _merge_tiles(self, line: np.ndarray) -> tuple[np.ndarray, int]:
        """Merge the tiles of a specific row or column"""
        non_zero_tiles = [tile for tile in line if tile != 0]
        merged_line = []
        skip = False
        n_tiles = len(non_zero_tiles)
        
        merge_score: int = 0
        for i in range(n_tiles):
            if skip:
                skip = False
                continue
            if i + 1 < n_tiles and non_zero_tiles[i] == non_zero_tiles[i + 1]:
                merged_tile = non_zero_tiles[i] * 2
                merge_score += merged_tile
                merged_line.append(merged_tile)
                skip = True
            else:
                merged_line.append(non_zero_tiles[i])
        
        merged_line += [0] * (len(line) - len(merged_line))
        return np.array(merged_line), merge_score

    def _move(self, grid: np.ndarray, action: int) -> tuple[np.ndarray, int]:
        """Move the tiles of the grid in a specific direction."""
        new_grid = np.zeros((self.size, self.size), dtype=int)
        
        merge_reward = 0
        
        if action == 0:  # Up
            for col in range(self.size):
                new_grid[:, col], reward = self._merge_tiles(grid[:, col])  # Merge column-wise
                merge_reward += reward
                
        elif action == 1:  # Down
            for col in range(self.size):
                merged_grid, reward = self._merge_tiles(np.flip(grid[:, col]))
                new_grid[:, col] = np.flip(merged_grid)  # Reverse merge for downward motion
                merge_reward += reward
                
        elif action == 2:  # Left
            for row in range(self.size):
                new_grid[row, :], reward = self._merge_tiles(grid[row, :])  # Merge row-wise
                merge_reward += reward

        elif action == 3:  # Right
            for row in range(self.size):
                merged_grid, reward = self._merge_tiles(np.flip(grid[row, :]))
                new_grid[row, :] = np.flip(merged_grid)  # Reverse merge for rightward motion
                merge_reward += reward
        else:
            raise ValueError("Invalid direction")
        
        return new_grid, merge_reward


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
    
    def is_win(self, grid: np.ndarray) -> bool:
        """Check if the player has won the game"""
        return np.any(grid == self.max_number)

    def step(self, old_grid: np.ndarray, action: int) -> tuple[np.ndarray, bool, int]:
        """Play a step of the game"""
        
        # Calculate the new grid after the move
        new_grid, self.merge_reward = self._move(old_grid, action)
        # Check if the game is over
        is_game_over = self.is_game_over(new_grid)
        
        # Check if the grid has changed
        if not np.array_equal(old_grid, new_grid):
            # If the grid has changed, add a new tile and check if the game is over
            new_grid = self._add_new_tile(new_grid)
        
        return new_grid, is_game_over, self.merge_reward