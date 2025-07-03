import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
from IPython.display import display, clear_output

# --- Configuration ---
GRID_SIZE = 50
TREE_DENSITY = 0.7
P_LIGHTNING = 0.00005
P_SPREAD = 0.1

EMPTY = 0
TREE = 1
BURNING = 2

colors = ['black', 'green', 'red']
cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

def initialize_forest(grid_size, tree_density):
    # The forest is a grid where each cell is either empty or has a tree.
    # In the beginning, the forest is peaceful and green.
    forest = np.random.choice([EMPTY, TREE], size=(grid_size, grid_size),
                              p=[1 - tree_density, tree_density])
    return forest

def update_forest(forest):
    # Each step, the fire spreads, trees may ignite from lightning, and burned trees become empty.
    new_forest = np.copy(forest)
    rows, cols = forest.shape
    for r in range(rows):
        for c in range(cols):
            current_state = forest[r, c]
            if current_state == EMPTY:
                new_forest[r, c] = EMPTY  # Ashes remain where fire once raged.
            elif current_state == BURNING:
                new_forest[r, c] = EMPTY  # Trees that burned down leave empty ground.
            elif current_state == TREE:
                # Sometimes, a bolt of lightning strikes, igniting a tree.
                if random.random() < P_LIGHTNING:
                    new_forest[r, c] = BURNING
                    continue
                fire_nearby = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if forest[nr, nc] == BURNING:
                                fire_nearby = True
                                break
                    if fire_nearby:
                        break
                # If fire is nearby, the tree may catch fire.
                if fire_nearby:
                    new_forest[r, c] = BURNING
                else:
                    new_forest[r, c] = TREE
    return new_forest

def simulate_forest_fire(generations=100):
    # The story begins: a dense forest, full of life.
    forest = initialize_forest(GRID_SIZE, TREE_DENSITY)
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(forest, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title("Forest Fire Simulation")
    ax.set_xticks([])
    ax.set_yticks([])

    for gen in range(generations):
        forest = update_forest(forest)
        img.set_data(forest)
        ax.set_title(f"Forest Fire Simulation - Generation {gen+1}/{generations}")
        display(fig)
        clear_output(wait=True)
        time.sleep(0.05)
        # The fire may die out, or the forest may be reduced to ashes.
        if not np.any(forest == BURNING) and not np.any(forest == TREE):
            print(f"Simulation ended early at generation {gen+1}: All trees have burned or no trees left to burn.")
            break
        elif not np.any(forest == BURNING) and P_LIGHTNING == 0:
            print(f"Simulation ended early at generation {gen+1}: All fire has died out and no new lightning strikes possible.")
            break
    plt.show()  # Show the final frame

def replay_simulation():
    # The forest can regrow, and the story can begin anew.
    while True:
        simulate_forest_fire(generations=200)
        print("\nSimulation complete!")
        replay = input("Replay the simulation? (y/n): ").strip().lower()
        if replay != 'y':
            print("Thank you for watching the story of the forest.")
            break

if __name__ == "__main__":
    # The cycle of fire and regrowth continues as long as you wish.
    replay_simulation()
