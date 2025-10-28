 import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------
# Parallel Cellular Algorithm - SIR Model
# -----------------------------------------
# States:
# 0 = Susceptible
# 1 = Infected
# 2 = Recovered
# -----------------------------------------

# Parameters
grid_size = 50            # Grid dimensions (50x50)
infection_prob = 0.3      # Probability an infected neighbor infects a susceptible cell
recovery_prob = 0.05      # Probability an infected cell recovers per iteration
initial_infected_frac = 0.01  # Initial infected fraction
steps = 100               # Number of time steps

# Initialize grid
grid = np.zeros((grid_size, grid_size), dtype=int)

# Randomly infect some cells
num_infected = int(initial_infected_frac * grid_size * grid_size)
infected_indices = np.random.choice(grid_size * grid_size, num_infected, replace=False)
grid.flat[infected_indices] = 1

# Define neighborhood (8 directions)
def infected_neighbors(grid):
    total = np.zeros_like(grid)
    shifts = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for dx, dy in shifts:
        total += np.roll(np.roll(grid == 1, dx, axis=0), dy, axis=1)
    return total

# Run simulation
for t in range(steps):
    infected_neigh = infected_neighbors(grid)

    # Infection step
    susceptible = (grid == 0)
    prob_infect = 1 - (1 - infection_prob)**infected_neigh
    random_vals = np.random.random(grid.shape)
    new_infections = susceptible & (random_vals < prob_infect)

    # Recovery step
    infected = (grid == 1)
    random_vals2 = np.random.random(grid.shape)
    new_recoveries = infected & (random_vals2 < recovery_prob)

    # Update states in parallel
    grid[new_infections] = 1
    grid[new_recoveries] = 2

    # Display
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.title(f"Disease Spread - Step {t}")
    plt.axis('off')
    plt.pause(0.1)
    plt.clf()

plt.imshow(grid, cmap='viridis', interpolation='nearest')
plt.title("Final State")
plt.axis('off')
plt.show()
