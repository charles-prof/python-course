import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Parameters
size = 50  # Grid size
p_tree = 0.6  # Probability of a tree
p_fire = 0.001  # Probability of spontaneous fire
frames = 100  # Number of frames

# States
EMPTY, TREE, FIRE = 0, 1, 2

# Initialize forest
def init_forest(size, p_tree):
    forest = np.random.choice([EMPTY, TREE], size=(size, size), p=[1-p_tree, p_tree])
    # Start fire in the middle
    forest[size//2, size//2] = FIRE
    return forest

def step(forest):
    new_forest = forest.copy()
    for i in range(forest.shape[0]):
        for j in range(forest.shape[1]):
            if forest[i, j] == TREE:
                # Check neighbors for fire
                neighbors = forest[max(i-1,0):i+2, max(j-1,0):j+2]
                if np.any(neighbors == FIRE):
                    new_forest[i, j] = FIRE
                elif np.random.rand() < p_fire:
                    new_forest[i, j] = FIRE
            elif forest[i, j] == FIRE:
                new_forest[i, j] = EMPTY
    return new_forest

# Visualization
cmap = mcolors.ListedColormap(['black', 'green', 'red'])
bounds = [EMPTY, TREE, FIRE, FIRE+1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

forest = init_forest(size, p_tree)
fig, ax = plt.subplots()
im = ax.imshow(forest, cmap=cmap, norm=norm)
ax.axis('off')

def update(frame):
    global forest
    forest = step(forest)
    im.set_data(forest)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()
