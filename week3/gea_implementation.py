import cv2
import numpy as np
import random

# GEA Parameters
POP_SIZE = 20
GENS = 15
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
X_MIN, X_MAX = 0, 255
GENES = 8  # Number of genes to represent the threshold

# Load grayscale image
img = cv2.imread('D:/1BM23CS145/BIS/download.jpg', 0)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Histogram of the image
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()

# Fitness function: Otsu-based threshold fitness
def fitness_function(threshold):
    threshold = int(threshold)

    # Compute weights
    w0 = np.sum(hist[:threshold])
    w1 = np.sum(hist[threshold:])

    # Avoid divide-by-zero
    if w0 == 0 or w1 == 0:
        return 0

    # Compute means
    mu0 = np.sum(np.arange(0, threshold) * hist[:threshold]) / w0
    mu1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1

    # Between-class variance (Otsu’s criterion)
    return w0 * w1 * ((mu0 - mu1) ** 2)

def create_individual():
    # Create an individual (chromosome) with a sequence of 8 bits, representing the threshold
    return ''.join(random.choice(['0', '1']) for _ in range(GENES))

def decode_individual(individual):
    # Decode binary string to threshold value
    return int(individual, 2)

# Tournament selection for parents
def select_parents(population):
    selected = random.sample(population, 3)
    selected.sort(key=lambda ind: fitness_function(decode_individual(ind)), reverse=True)
    return selected[0], selected[1]

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, GENES - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
    else:
        child = parent1
    return child

def mutate(individual):
    if random.random() < MUTATION_RATE:
        mutation_point = random.randint(0, GENES - 1)
        mutated_individual = list(individual)
        mutated_individual[mutation_point] = '1' if mutated_individual[mutation_point] == '0' else '0'
        individual = ''.join(mutated_individual)
    return individual

def genetic_algorithm():
    # Initialize population with random individuals (binary strings)
    population = [create_individual() for _ in range(POP_SIZE)]
    best_solution = max(population, key=lambda ind: fitness_function(decode_individual(ind)))

    for generation in range(GENS):
        new_population = []
        for _ in range(POP_SIZE):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        current_best = max(population, key=lambda ind: fitness_function(decode_individual(ind)))
        if fitness_function(decode_individual(current_best)) > fitness_function(decode_individual(best_solution)):
            best_solution = current_best

        print(f"Generation {generation + 1}: Best Threshold (decoded) = {decode_individual(best_solution)}")

    print(f"\nBest threshold found: {decode_individual(best_solution)}")

    # Apply the threshold and display/save result
    best_threshold = decode_individual(best_solution)
    _, segmented_img = cv2.threshold(img, best_threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite("segmented_gea.jpg", segmented_img)
    print("Segmented image saved as 'segmented_gea.jpg'.")

if __name__ == "__main__":
    genetic_algorithm()
