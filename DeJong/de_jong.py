import random
import math
import time

MAXIMUM_GENERATION_NUMBER = 5000
POPULATION_SIZE = 100
INITIAL_MUTATION_PROBABILITY = 0.01
INITIAL_CROSSOVER_PROBABILITY = 0.9
TOURNAMENT_SIZE = 5
NO_IMPROVEMENT_LIMIT = 500

n = 10
d = 5

a, b, global_minima = None, None, None
func = None

def initialize_specifications():
    global N, l, L
    N = int((b - a) * 10**d)
    l = math.ceil(math.log2(N))
    L = l * n

def DeJong(x):
    return sum(xi**2 for xi in x)

def Schwefel(x):
    return 418.9829 * n - sum(abs(xi) * math.sin(math.sqrt(abs(xi))) for xi in x)

def generate_bitstring():
    return ''.join(str(random.randint(0, 1)) for _ in range(l))

def generate_vector_bitstring():
    return [generate_bitstring() for _ in range(n)]

def decode_bitstring(bitstring):
    value = int(bitstring, 2)
    return a + value * (b - a) / (2**l - 1)

def decode_vector(vector_bitstring):
    return [decode_bitstring(bitstring) for bitstring in vector_bitstring]

def evaluate_candidate(vector_bitstring):
    decoded_vector = decode_vector(vector_bitstring)
    if func == 'j':
        return DeJong(decoded_vector)
    elif func == 's':
        return Schwefel(decoded_vector)
    return None

def generate_population():
    return [generate_vector_bitstring() for _ in range(POPULATION_SIZE)]

def evaluate(population):
    return [evaluate_candidate(ind) for ind in population]

def tournament_selection(population, fitness):
    new_population = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(range(len(population)), TOURNAMENT_SIZE)
        winner = min(tournament, key=lambda idx: fitness[idx])
        new_population.append(population[winner])
    return new_population

def mutate(population, mutation_probability):
    for i in range(len(population)):
        for j in range(n):
            if random.random() < mutation_probability:
                k = random.randint(0, l - 1)
                population[i][j] = population[i][j][:k] + ('1' if population[i][j][k] == '0' else '0') + population[i][j][k+1:]
    return population

def crossover(population, crossover_probability):
    new_population = []
    for i in range(0, len(population) - 1, 2):
        if random.random() < crossover_probability:
            parent1 = ''.join(population[i])
            parent2 = ''.join(population[i + 1])
            crossover_point = random.randint(1, len(parent1) - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            new_population.append([offspring1[k:k + l] for k in range(0, len(offspring1), l)])
            new_population.append([offspring2[k:k + l] for k in range(0, len(offspring2), l)])
        else:
            new_population.extend([population[i], population[i + 1]])
    return new_population

def genetic_algorithm():
    global a, b, func, global_minima
    a, b = -500, 500
    func = 's'
    initialize_specifications()

    population = generate_population()
    best_solution = None
    best_fitness = float('inf')
    no_improvement_counter = 0
    mutation_probability = INITIAL_MUTATION_PROBABILITY
    crossover_probability = INITIAL_CROSSOVER_PROBABILITY

    for generation in range(MAXIMUM_GENERATION_NUMBER):
        fitness = evaluate(population)
        best_index = fitness.index(min(fitness))
        if fitness[best_index] < best_fitness:
            best_fitness = fitness[best_index]
            best_solution = population[best_index]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= NO_IMPROVEMENT_LIMIT:
            break

        elitism = [population[best_index]]
        population = tournament_selection(population, fitness)
        population = crossover(population, crossover_probability)
        population = mutate(population, mutation_probability)
        population = elitism + population[:-1]

        # Gradually reduce mutation and crossover probabilities
        mutation_probability *= 0.99
        crossover_probability *= 0.99

    global_minima = decode_vector(best_solution)
    return global_minima, best_fitness

fitness_values = []

for _ in range(30):
    solution, fitness = genetic_algorithm()
    fitness_values.append(fitness)

min_fitness = min(fitness_values)
max_fitness = max(fitness_values)
avg_fitness = sum(fitness_values) / len(fitness_values)

print(f"Min Fitness: {min_fitness:.5f}")
print(f"Max Fitness: {max_fitness:.5f}")
print(f"Avg Fitness: {avg_fitness:.5f}")
