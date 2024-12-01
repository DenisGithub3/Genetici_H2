import numpy as np
import time
from typing import Tuple , List

class BinaryGA:
    def __init__(self, bits_per_param: int = 16, 
                 n_params: int = 30,  # DIMENSIUNI 
                 bounds: Tuple[float, float] = (0, np.pi)):
        self.bits_per_param = bits_per_param
        self.n_params = n_params
        self.string_length = bits_per_param * n_params
        self.bounds = bounds
        # Precompute powers of two for decoding
        self.powers_of_two = 2 ** np.arange(self.bits_per_param - 1, -1, -1)
        self.largest = 2 ** self.bits_per_param - 1
        
    def create_population(self, pop_size: int) -> np.ndarray:
        """Generate initial population as a 2D NumPy array."""
        return np.random.randint(0, 2, (pop_size, self.string_length))
    
    def decode(self, bitstrings: np.ndarray) -> np.ndarray:
        """Decode bitstrings to real values."""
        # Reshape bitstrings to (pop_size, n_params, bits_per_param)
        pop_size = bitstrings.shape[0]
        bit_array = bitstrings.reshape(pop_size, self.n_params, self.bits_per_param)
        # Convert binary to decimal
        integers = bit_array.dot(self.powers_of_two)
        # Scale to bounds [0, pi]
        values = self.bounds[0] + (integers / self.largest) * (self.bounds[1] - self.bounds[0])
        return values  # shape: (pop_size, n_params)
    
    def michalewicz(self, x: np.ndarray, m: int = 10) -> np.ndarray:
        """Compute Michalewicz function for multiple candidates."""
        i = np.arange(1, self.n_params + 1)
        # Reshape i for broadcasting
        i = i[np.newaxis, :]
        sin_x = np.sin(x)
        sin_term = np.sin((x ** 2) * i / np.pi) ** (2 * m)
        return -np.sum(sin_x * sin_term, axis=1)  # Return array of shape (pop_size,)
    
    def selection(self, pop: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Tournament selection with tournament size of 3."""
        pop_size = pop.shape[0]
        tournament_size = 3
        # Randomly select tournament candidates
        candidates_idx = np.random.randint(0, pop_size, size=(pop_size, tournament_size))
        # Get scores of candidates
        candidates_scores = scores[candidates_idx]  # shape (pop_size, tournament_size)
        # Find the index of the best candidate in each tournament
        best_candidate_indices = np.argmin(candidates_scores, axis=1)
        # Get indices of the selected individuals
        selected_indices = candidates_idx[np.arange(pop_size), best_candidate_indices]
        # Return selected individuals
        return pop[selected_indices]
    
    def crossover(self, parents: np.ndarray, r_cross: float) -> np.ndarray:
        """Apply crossover on a set of parents to produce children."""
        pop_size = parents.shape[0]
        children = parents.copy()
        for i in range(0, pop_size, 2):
            if np.random.rand() < r_cross:
                crossover_point = np.random.randint(1, self.string_length)
                children[i, crossover_point:], children[i+1, crossover_point:] = \
                    children[i+1, crossover_point:], children[i, crossover_point:]
        return children
    
    def mutation(self, bitstrings: np.ndarray, r_mut: float) -> np.ndarray:
        """Bit flip mutation applied to a population."""
        mutation_mask = np.random.rand(*bitstrings.shape) < r_mut
        bitstrings[mutation_mask] = 1 - bitstrings[mutation_mask]
        return bitstrings
    
    def run(self, n_iter: int = 50000, r_cross: float = 0.9, r_mut: float = 1.0/100) -> Tuple[np.ndarray, float]:
        """Run the genetic algorithm."""
        # Initial population
        pop_size = 100
        pop = self.create_population(pop_size)
        best, best_eval = None, float('inf')
        
        for gen in range(n_iter):
            # Decode and evaluate all candidates
            decoded = self.decode(pop)  # shape: (pop_size, n_params)
            scores = self.michalewicz(decoded)  # shape: (pop_size,)
            
            # Check for new best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_eval:
                best, best_eval = pop[min_idx].copy(), scores[min_idx]
                #print(f'>Gen {gen}: new best = {best_eval}')
            
            # Implement elitism: Keep the best individual
            elite = pop[min_idx].copy()
            
            # Select parents
            selected = self.selection(pop, scores)
            
            # Apply crossover
            children = self.crossover(selected, r_cross)
            
            # Apply mutation
            children = self.mutation(children, r_mut)
            
            # Replace population with children
            pop = children
            
            # Apply elitism: Replace a random individual with the elite
            replace_idx = np.random.randint(pop_size)
            pop[replace_idx] = elite
        
        return best, best_eval

# Usage example
if __name__ == "__main__":
    start_time = time.time()
    ga = BinaryGA()
    best_bitstring, score = ga.run()
    decoded = ga.decode(np.array([best_bitstring]))[0]  # Decode the best individual
    end_time = time.time()
    #print('Done!')
    print(f'Rezultat : {score:.5f}')
    running_time = end_time - start_time
    print(f"Running time : {running_time:.5f} seconds")
    print(f"\n")