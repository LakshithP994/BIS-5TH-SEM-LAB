import numpy as np

def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

num_particles = 30
num_dimensions = 2
num_iterations = 55
w = 0.5             # Inertia weight
c1 = 1.5            # Cognitive coefficient
c2 = 1.5            # Social coefficient
bounds = (-5.12, 5.12)

positions = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
personal_best_positions = positions.copy()
personal_best_scores = np.array([rastrigin(p) for p in positions])


global_best_index = np.argmin(personal_best_scores)
global_best_position = personal_best_positions[global_best_index].copy()
global_best_score = personal_best_scores[global_best_index]

for iteration in range(num_iterations):
    for i in range(num_particles):
        
        r1 = np.random.rand(num_dimensions)
        r2 = np.random.rand(num_dimensions)
        cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
        social = c2 * r2 * (global_best_position - positions[i])
        velocities[i] = w * velocities[i] + cognitive + social

        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        score = rastrigin(positions[i])

        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

        if score < global_best_score:
            global_best_score = score
            global_best_position = positions[i]

    print(f"Iteration {iteration+1}/{num_iterations}, Best Score: {global_best_score:.5f}")

print("\n=== Optimization Result ===")
print("Best Position:", global_best_position)
print("Best Score:", global_best_score)
