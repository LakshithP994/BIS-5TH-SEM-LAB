import numpy as np

stars_data = np.array([
    [5778, 1.0, 1.0, 1.0],  # T, R, M, L (for the Sun)
    [6000, 1.2, 1.1, 1.2],  # Another star example
    [5000, 0.8, 0.9, 0.5],
    [6500, 1.3, 1.4, 1.4],
])

def predict_luminosity(a, b, c, d, star_data):
    T, R, M = star_data[:, 0], star_data[:, 1], star_data[:, 2]
    L_pred = a * R**b * T**c * M**d
    return L_pred

def objective_function(params, data):
    a, b, c, d = params
    actual_L = data[:, 3]  
    predicted_L = predict_luminosity(a, b, c, d, data)
    mse = np.mean((predicted_L - actual_L) ** 2)  
    return mse

N = 30  # Number of particles
D = 4   # Number of parameters (a, b, c, d)
T_max = 10  # Number of iterations
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

positions = np.random.uniform(-10, 10, (N, D))  
velocities = np.random.uniform(-1, 1, (N, D))  
personal_best_positions = positions.copy()
personal_best_scores = np.array([objective_function(p, stars_data) for p in positions])

global_best_index = np.argmin(personal_best_scores)
global_best_position = personal_best_positions[global_best_index].copy()
global_best_score = personal_best_scores[global_best_index]

for t in range(T_max):
    for i in range(N):
        r1 = np.random.rand(D)
        r2 = np.random.rand(D)
        velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - positions[i]) + c2 * r2 * (global_best_position - positions[i])

        positions[i] += velocities[i]
        
        score = objective_function(positions[i], stars_data)

        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

        if score < global_best_score:
            global_best_score = score
            global_best_position = positions[i]

    print(f"Iteration {t+1}/{T_max}, Best Score: {global_best_score:.5f}")

print("\n=== Optimization Result ===")
print(f"Best Parameters: a={global_best_position[0]:.5f}, b={global_best_position[1]:.5f}, c={global_best_position[2]:.5f}, d={global_best_position[3]:.5f}")
print(f"Best Score (MSE): {global_best_score:.5f}")
