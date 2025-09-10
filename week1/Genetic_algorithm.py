import random
import math

# Objective Function
def fitness_function(x):
    return x * math.sin(10 * math.pi * x) + 1 #write the matematical function here as needed

# GA Parameters ; initialising parameters
POP_SIZE = 20
GENS = 50
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
X_MIN, X_MAX = 0, 1

# Generate random individual
def create_individual():
    return random.uniform(X_MIN, X_MAX)

# Selection: Tournament Selection
def select_parents(population):
    selected = random.sample(population, 3)
    selected.sort(key=fitness_function, reverse=True)
    return selected[0], selected[1]

# Crossover: Blend Crossover (BLX-alpha)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        alpha = 0.5
        d = abs(parent1 - parent2)
        low = max(X_MIN, min(parent1, parent2) - alpha * d)
        high = min(X_MAX, max(parent1, parent2) + alpha * d)
        return random.uniform(low, high)
    else:
        return parent1

# Mutation
def mutate(individual):
    if random.random() < MUTATION_RATE:
        mutation_amount = random.uniform(-0.1, 0.1)
        individual += mutation_amount
        individual = max(min(individual, X_MAX), X_MIN)
    return individual

def genetic_algorithm():
    population = [create_individual() for _ in range(POP_SIZE)]
    best_solution = max(population, key=fitness_function)

    for generation in range(GENS):
        new_population = []
        for _ in range(POP_SIZE):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        current_best = max(population, key=fitness_function)
        if fitness_function(current_best) > fitness_function(best_solution):
            best_solution = current_best

        print(f"Generation {generation+1}: Best = {fitness_function(best_solution):.5f}")

    print("\nBest solution found:")
    print(f"x = {best_solution:.5f}, f(x) = {fitness_function(best_solution):.5f}")

genetic_algorithm()
