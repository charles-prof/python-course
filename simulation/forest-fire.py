# Sample input data for the forest fire cellular automaton
# Each cell is either:
# 0 - EMPTY (black), 1 - TREE (green), 2 - FIRE (red)
# This is a small 5x5 grid for demonstration.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

sample_forest = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 2, 1, 1],
    [1, 2, 1, 2, 1],
    [1, 1, 2, 1, 1],
    [0, 1, 1, 1, 0]
])

# Forest Fire Cellular Automaton Implementation

# Parameters
size = 50  # Grid size
p_tree = 0.6  # Probability of a tree
p_fire = 0.001  # Probability of spontaneous fire
frames = 100  # Number of frames

# States
EMPTY, TREE, FIRE = 0, 1, 2

def init_forest(size, p_tree):
    """
    Initialize the forest grid with trees and empty cells.
    The center cell is set on fire.
    """
    forest = np.random.choice([EMPTY, TREE], size=(size, size), p=[1-p_tree, p_tree])
    forest[size//2, size//2] = FIRE  # Ignite the center
    return forest

def get_neighbors(grid, x, y):
    """
    Get the 8 neighbors of cell (x, y) in the grid.
    Handles edge cases by limiting indices.
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the cell itself
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                neighbors.append(grid[nx, ny])
    return neighbors

def ca_step(forest):
    """
    Perform one step of the cellular automaton.
    Rules:
    - A TREE becomes FIRE if at least one neighbor is FIRE or with probability p_fire.
    - A FIRE becomes EMPTY.
    - EMPTY remains EMPTY.
    """
    new_forest = forest.copy()
    for i in range(forest.shape[0]):
        for j in range(forest.shape[1]):
            cell = forest[i, j]
            if cell == TREE:
                neighbors = get_neighbors(forest, i, j)
                if FIRE in neighbors or np.random.rand() < p_fire:
                    new_forest[i, j] = FIRE
            elif cell == FIRE:
                new_forest[i, j] = EMPTY
            # EMPTY stays EMPTY
    return new_forest

# Visualization setup
cmap = mcolors.ListedColormap(['black', 'green', 'red'])
bounds = [EMPTY, TREE, FIRE, FIRE+1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

forest = init_forest(size, p_tree)
fig, ax = plt.subplots()
im = ax.imshow(forest, cmap=cmap, norm=norm)
ax.axis('off')

def update(frame):
    """
    Animation update function for each frame.
    """
    global forest
    forest = ca_step(forest)
    im.set_data(forest)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()
