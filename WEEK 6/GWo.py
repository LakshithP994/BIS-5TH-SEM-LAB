import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# PV Model Parameters
# -----------------------
Iph = 5.5      # Photocurrent (A)
Io = 1e-6      # Saturation current (A)
Rs = 0.2       # Series resistance (Ohm)
Rp = 100       # Parallel resistance (Ohm)
a = 1.3        # Diode ideality factor
Vt = 0.026     # Thermal voltage (V)

# PV current equation
def pv_current(V):
    return Iph - Io * (np.exp((V + (Rs * Iph)) / (a * Vt)) - 1) - (V + Rs * Iph) / Rp

def power_output(V):
    I = pv_current(V)
    return V * I

# -----------------------
# Gray Wolf Optimization (GWO)
# -----------------------
def GWO(obj_func, lb, ub, dim, N, max_iter):
    # Initialize positions
    X = np.random.uniform(lb, ub, (N, dim))
    Alpha = np.zeros(dim)
    Beta = np.zeros(dim)
    Delta = np.zeros(dim)
    Alpha_score = Beta_score = Delta_score = -np.inf

    convergence_curve = []

    for t in range(max_iter):
        for i in range(N):
            fitness = obj_func(X[i, :])
            if fitness > Alpha_score:
                Delta_score = Beta_score
                Delta = Beta.copy()
                Beta_score = Alpha_score
                Beta = Alpha.copy()
                Alpha_score = fitness
                Alpha = X[i, :].copy()
            elif fitness > Beta_score:
                Delta_score = Beta_score
                Delta = Beta.copy()
                Beta_score = fitness
                Beta = X[i, :].copy()
            elif fitness > Delta_score:
                Delta_score = fitness
                Delta = X[i, :].copy()

        a = 2 - t * (2 / max_iter)
        for i in range(N):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha[j] - X[i, j])
                X1 = Alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta[j] - X[i, j])
                X2 = Beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta[j] - X[i, j])
                X3 = Delta[j] - A3 * D_delta

                X[i, j] = (X1 + X2 + X3) / 3

            X[i, :] = np.clip(X[i, :], lb, ub)

        convergence_curve.append(Alpha_score)

    return Alpha, Alpha_score, convergence_curve

# -----------------------
# Run Simulation
# -----------------------
lb, ub = 0, 0.6    # Voltage limits (V)
dim = 1
N = 20
max_iter = 50

best_V, best_P, curve = GWO(power_output, lb, ub, dim, N, max_iter)

# Compute corresponding current
best_I = pv_current(best_V)[0]

print(f"Optimal Voltage (Vmp): {best_V[0]:.4f} V")
print(f"Optimal Current (Imp): {best_I:.4f} A")
print(f"Maximum Power (Pmax): {best_P:.4f} W")

# -----------------------
# Visualization
# -----------------------
V = np.linspace(0, 0.6, 100)
P = power_output(V)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(V, P, label='P–V Curve')
plt.scatter(best_V, best_P, color='r', label='GWO MPP', zorder=5)
plt.title("PV Power–Voltage Curve")
plt.xlabel("Voltage (V)")
plt.ylabel("Power (W)")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(curve, 'b-', label='Convergence Curve')
plt.title("GWO Convergence Behavior")
plt.xlabel("Iteration")
plt.ylabel("Power (W)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
