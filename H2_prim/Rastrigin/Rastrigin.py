import numpy as np
import time
from typing import Tuple

class BinaryGA:
    def __init__(self, bits_per_param: int = 16,
                 n_params: int = 30,  # DIMENSIUNI
                 bounds: Tuple[float, float] = (-500, 500)):
        self.bits_per_param = bits_per_param
        self.n_params = n_params
        self.string_length = bits_per_param * n_params
        self.bounds = bounds
        # Precalculăm puterile lui doi pentru decodificare
        self.powers_of_two = 2 ** np.arange(self.bits_per_param - 1, -1, -1)
        # Creează un array descrescător pentru puterile lui 2 
        self.largest = 2 ** self.bits_per_param - 1

    def create_population(self, pop_size: int) -> np.ndarray:
        """Generează populația inițială ca un array 2D NumPy."""
        return np.random.randint(0, 2, (pop_size, self.string_length))

    def gray_to_binary(self, gray_bits: np.ndarray) -> np.ndarray:
        """Convertim șirurile de biți din codul Gray în cod binar."""
        binary_bits = np.zeros_like(gray_bits)
        binary_bits[..., 0] = gray_bits[..., 0]
        for i in range(1, self.bits_per_param):
            binary_bits[..., i] = np.bitwise_xor(binary_bits[..., i - 1], gray_bits[..., i])
        return binary_bits

    def decode(self, bitstrings: np.ndarray) -> np.ndarray:
        """Decodifică șirurile de biți în valori reale."""
        # Reshape la (pop_size, n_params, bits_per_param)
        pop_size = bitstrings.shape[0]
        bit_array = bitstrings.reshape(pop_size, self.n_params, self.bits_per_param)
        # Convertim codul Gray în cod binar
        binary_bits = self.gray_to_binary(bit_array)
        # Convertim binarul în zecimal
        integers = binary_bits.dot(self.powers_of_two)
        # Scalăm la limitele [-500, 500]
        values = self.bounds[0] + (integers / self.largest) * (self.bounds[1] - self.bounds[0])
        return values  # formă: (pop_size, n_params)

    def rastrigin(self, x: np.ndarray) -> np.ndarray:
        """Compute Rastrigin function for multiple candidates."""
        n = self.n_params
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
    
    def selection(self, pop: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Selecție turnir cu dimensiunea turnirului de 3."""
        pop_size = pop.shape[0]
        tournament_size = 3
        # Selectăm aleatoriu candidații pentru turnir
        candidates_idx = np.random.randint(0, pop_size, size=(pop_size, tournament_size))
        # Obținem scorurile candidaților
        candidates_scores = scores[candidates_idx]  # formă (pop_size, tournament_size)
        # Găsim indicele celui mai bun candidat în fiecare turnir
        best_candidate_indices = np.argmin(candidates_scores, axis=1)
        # Obținem indicii indivizilor selectați
        selected_indices = candidates_idx[np.arange(pop_size), best_candidate_indices]
        # Returnăm indivizii selectați
        return pop[selected_indices]

    def crossover(self, parents: np.ndarray, r_cross: float) -> np.ndarray:
        """Aplică încrucișarea pe un set de părinți pentru a produce copii."""
        pop_size = parents.shape[0]
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < r_cross:
                crossover_point = np.random.randint(1, self.string_length)
                children[i, crossover_point:], children[i+1, crossover_point:] = \
                    children[i+1, crossover_point:], children[i, crossover_point:]
        return children

    def mutation(self, bitstrings: np.ndarray, r_mut: float) -> np.ndarray:
        """Mutarea prin flip de bit aplicată unei populații."""
        mutation_mask = np.random.rand(*bitstrings.shape) < r_mut
        bitstrings[mutation_mask] = 1 - bitstrings[mutation_mask]
        return bitstrings

    def run(self, n_iter: int = 50000, r_cross: float = 0.9, r_mut: float = 1.0 / 100) -> Tuple[np.ndarray, float]:
        """Rulează algoritmul genetic."""
        # Populația inițială
        pop_size = 100
        pop = self.create_population(pop_size)
        best, best_eval = None, float('inf')

        for gen in range(n_iter):
            # Decodificăm și evaluăm toți candidații
            decoded = self.decode(pop)  # formă: (pop_size, n_params)
            scores = self.rastrigin(decoded)  # formă: (pop_size,)

            # Verificăm pentru un nou cel mai bun
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_eval:
                best, best_eval = pop[min_idx].copy(), scores[min_idx]
                #print(f'>Gen {gen}: new best = {best_eval}')

            # Implementăm elitismul: Păstrăm cel mai bun individ
            elite = pop[min_idx].copy()

            # Selectăm părinții
            selected = self.selection(pop, scores)

            # Aplicăm încrucișarea
            children = self.crossover(selected, r_cross)

            # Aplicăm mutația
            children = self.mutation(children, r_mut)

            # Înlocuim populația cu copiii
            pop = children

            # Aplicăm elitismul: Înlocuim un individ aleatoriu cu elita
            replace_idx = np.random.randint(pop_size)
            pop[replace_idx] = elite

        return best, best_eval

# Exemplu de utilizare
if __name__ == "__main__":
    start_time = time.time()
    ga = BinaryGA()
    best_bitstring, score = ga.run()
    decoded = ga.decode(np.array([best_bitstring]))[0]  # Decodificăm cel mai bun individ
    end_time = time.time()
    # print('Gata!')
    # print(f'Cel mai bun: f({decoded}) = {score}')
    print(f'Rezultat : {score:.5f}')
    running_time = end_time - start_time
    print(f"Timp de rulare: {running_time:.5f} secunde")
    print("\n")
