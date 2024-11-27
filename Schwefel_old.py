import numpy as np
from typing import Tuple, List
import time

class BinaryGA:
    def __init__(self, bits_per_param: int = 16, 
                 n_params: int = 30,  # DIMENSIUNI
                 bounds: Tuple[float, float] = (-500, 500)):
        self.bits_per_param = bits_per_param
        self.n_params = n_params
        self.string_length = bits_per_param * n_params
        self.bounds = bounds
        # Precompute powers of two for decoding
        self.powers_of_two = 2 ** np.arange(self.bits_per_param)[::-1]
        self.largest = 2 ** self.bits_per_param - 1
        
    def create_bitstring(self) -> np.ndarray:
        """Generate random bitstring as numpy array."""
        return np.random.randint(0, 2, self.string_length)
    
    def decode(self, bitstring: np.ndarray) -> np.ndarray:
        """Decode bitstring to real values."""
        decoded = []
        for i in range(self.n_params):
            # Extract substring
            start = i * self.bits_per_param
            end = start + self.bits_per_param
            substring = bitstring[start:end]
            # Convert to decimal
            integer = substring.dot(self.powers_of_two)
            # Scale to bounds
            value = self.bounds[0] + (integer / self.largest) * (self.bounds[1] - self.bounds[0])
            decoded.append(value)
        return np.array(decoded)
    
    def schwefel(self, x: np.ndarray) -> float:
        """Compute Schwefel function."""
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def crossover(self, p1: np.ndarray, p2: np.ndarray, r_cross: float) -> Tuple[np.ndarray, np.ndarray]:
        """Single point crossover."""
        c1, c2 = p1.copy(), p2.copy()
        if np.random.random() < r_cross:
            point = np.random.randint(1, len(p1))
            c1 = np.concatenate((p1[:point], p2[point:]))
            c2 = np.concatenate((p2[:point], p1[point:]))
        return c1, c2
    
    def mutation(self, bitstring: np.ndarray, r_mut: float) -> np.ndarray:
        """Bit flip mutation."""
        mutation_mask = np.random.rand(len(bitstring)) < r_mut
        bitstring[mutation_mask] = 1 - bitstring[mutation_mask]
        return bitstring
    
    def selection(self, pop: List[np.ndarray], scores: List[float]) -> np.ndarray:
        """Tournament selection with tournament size of 3."""
        selection_ix = np.random.randint(len(pop))
        for ix in np.random.randint(0, len(pop), 2):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]
    
    def run(self, n_iter: int = 5000, r_cross: float = 0.9, r_mut: float = 1.0/100) -> Tuple[np.ndarray, float]:
        """Run the genetic algorithm."""
        # Initial population of 100
        pop = [self.create_bitstring() for _ in range(100)]
        best, best_eval = None, float('inf')
        
        for gen in range(n_iter):
            # Decode and evaluate all candidates
            decoded = [self.decode(p) for p in pop]
            scores = [self.schwefel(d) for d in decoded]
            
            # Check for new best
            for i in range(len(scores)):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    #print(f'>Gen {gen}: new best = {best_eval}')
            
            # Select parents
            selected = [self.selection(pop, scores) for _ in range(len(pop))]
            
            # Create next generation
            children = []
            for i in range(0, len(selected), 2):
                # Get selected parents
                p1, p2 = selected[i], selected[i+1]
                # Crossover and mutation
                c1, c2 = self.crossover(p1, p2, r_cross)
                c1 = self.mutation(c1, r_mut)
                c2 = self.mutation(c2, r_mut)
                # Store for next generation
                children.extend([c1, c2])
            pop = children
            
        return best, best_eval

# Usage example
if __name__ == "__main__":
    start_time = time.time()
    ga = BinaryGA()
    best_bitstring, score = ga.run()
    decoded = ga.decode(best_bitstring)
    end_time = time.time()
    print('Done!')
    print(f'Best: f({decoded}) = {score}')
    print(f"Running time : {end_time - start_time} seconds")
