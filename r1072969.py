# Minimal evolutionary algorithm for directed TSP with possible ∞ (no-edge) distances.
import Reporter
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from collections import Counter
import random
import time

#import psutil
#import os
#import time


class r1072969:
    def __init__(self,
                 rng_seed: int | None = None,
                 mu: int = 100,                         # population size (optional: auto if None) ¿?
                 lamb: int = 100,                       # offspring per generation (optional: auto if None)
                 k_tournament: int = 3,                 # tournament size (optional: auto if None)
                 mutation_rate: float = 0.9,       # probability to mutate a selected parent ¿?
                 crossover_rate: float = 0.9,      # probability to recombine two parents ¿?
                 feasible_retry: int = 20          # retries to obtain a feasible variation
                 ):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.rng = np.random.default_rng(rng_seed)

        self.mu = mu
        self.lamb = lamb
        self.k_tournament = k_tournament
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.feasible_retry = feasible_retry
        
    # ---- Initialisation ----
    def initialisation(self, distanceMatrix: np.ndarray) -> np.ndarray:
        """ 
            Generate mu individuals (40% greedy, 60% random), all feasible if possible 
        """
        n = distanceMatrix.shape[0]
        M = self._finite_outgoing_mask(distanceMatrix)
        mu = self.mu
   
        population: list[np.ndarray] = []

        num_greedy_individuals = max(1, int(round(0.99 * mu))) # ensure at least 1 greedy
    
        greedy_tries, random_tries = 0, 0
        max_total_tries = 10_000
        
        # greedy tours (~40% population)
        while len(population) < num_greedy_individuals and (greedy_tries + random_tries) < max_total_tries:
            greedy_tries += 1 
            try:
                tour = self._randomised_greedy_initialisation_feasible(distanceMatrix, M)
                population.append(tour)
            except ValueError:
                continue

        # random tours (~60% population)
        while len(population) < mu and (greedy_tries + random_tries) < max_total_tries:
            random_tries += 1
            try:
                tour = self._randomised_initialisation_feasible(distanceMatrix, M)
                population.append(tour)
            except ValueError:
                continue

        # in case of ValueError raised, fill the rest with pure random and potentially infeasible permutations
        while len(population) < mu:
            population.append(self.rng.permutation(n))
        
        return population
            
    # ---- Selection ----
    def selection(self, population: list[np.ndarray], fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ 
            Compute k-tournament selection to pick two parents 
        """
        k = self.k_tournament
        mu = self.mu
        # Parent selection (k-tournament)
        p1_idx = self._tournament_select_idx(fitness, k)
        p2_idx = self._tournament_select_idx(fitness, k)
        while p2_idx == p1_idx and mu > 1:
            p2_idx = self._tournament_select_idx(fitness, k)
        p1, p2 = population[p1_idx], population[p2_idx]

        # Ensure p1 is the better parent (helps in fallback decisions)
        if fitness[p2_idx] < fitness[p1_idx]:
            p1, p2 = p2, p1
        return p1, p2
    
    # ---- Crossover ----
    def crossover(self, p1: np.ndarray, p2: np.ndarray, distanceMatrix: np.ndarray) -> np.ndarray:
        """ 
            Generate crossover_rate*lambda offsprings, the rest are copies of one parent
        """
        if self.rng.random() < self.crossover_rate:
            child = self._recombine_feasible(p1, p2, distanceMatrix)
        else:
            child = p1.copy()

        return child

    # ---- Mutation ----
    def mutation(self, child: np.ndarray, distanceMatrix: np.ndarray) -> np.ndarray:
        """ 
            Mutate mutation_rate*lambda of the children 
        """
        if self.rng.random() < self.mutation_rate:
            child = self._mutate_feasible(child, distanceMatrix)
        return child
    
    # --- Local Search Operator ---
    def local_search(self, offspring: list[np.ndarray], distanceMatrix: np.ndarray) -> list[np.ndarray]:
        """ 
            2-Opt local search on offspring (directed TSP, supports ∞ entries).
            First-improvement: for each child, repeatedly apply any improving 2-opt reversal
            until no improvement is found. Returns the improved offspring list.
        """
        improved_offspring: list[np.ndarray] = []

        # --- Budget ---
        p_ls = 0.8               # apply LS only to ~35% of children (set to 1.0 for all)
        max_evals = 4000          # max (i,k) evaluations per child
        max_moves = 120           # max improving reversals applied per child
        time_cap_ms = 10          # per-child time cap in milliseconds
        EPS = 1e-12

        def atsp_two_opt_delta_and_feasible(tour: np.ndarray, i: int, k: int) -> tuple[float, bool]:
            """
            Compute the change in tour length if we reverse segment (i+1 .. k),
            and check that all new arcs are finite. Works for directed/asymmetric matrices.

            Returns (delta, feasible_new_tour).
            """
            n = tour.size
            # adjacency including wrap: reversing a length-1 segment or (0, n-1) is a no-op
            if (i + 1) % n == k or (k + 1) % n == i:
                return 0.0, False

            a = tour[i]
            b = tour[(i + 1) % n]
            c = tour[k]
            d = tour[(k + 1) % n]

            # New endpoint arcs must be finite: a->c and b->d
            if not (np.isfinite(distanceMatrix[a, c]) and np.isfinite(distanceMatrix[b, d])):
                return 0.0, False

            # Start with endpoint changes
            delta = -distanceMatrix[a, b] - distanceMatrix[c, d] + distanceMatrix[a, c] + distanceMatrix[b, d]

            # Internal arcs reverse direction: (v_t -> v_{t+1}) becomes (v_{t+1} -> v_t)
            # Check finiteness and accumulate delta for each flipped arc
            # Segment (i+1 .. k) includes arcs (t, t+1) for t in [i+1 .. k-1]
            for t in range(i + 1, k):
                u = tour[t]
                v = tour[t + 1]
                # New arc will be v->u; must be finite
                if not np.isfinite(distanceMatrix[v, u]):
                    return 0.0, False
                delta += (distanceMatrix[v, u] - distanceMatrix[u, v])

            return float(delta), True

        def apply_two_opt_inplace(tour: np.ndarray, i: int, k: int) -> None:
            """Reverse the segment (i+1 .. k) in place; assumes i < k."""
            if i > k:
                i, k = k, i
            tour[i + 1 : k + 1] = tour[i + 1 : k + 1][::-1]

        for child in offspring:
            # Probabilistic LS to control runtime
            if self.rng.random() >= p_ls:
                improved_offspring.append(child)
                continue

            t = child.copy()
            # Skip clearly infeasible tours
            if not np.isfinite(self._tour_length(t, distanceMatrix)):
                improved_offspring.append(t)
                continue

            n = t.size
            start = time.perf_counter()
            evals = 0
            moves = 0
            
            improved = True
            while improved:
                improved = False
                # First-improvement sweep with budgets
                for i in range(n):
                    for k in range(i + 2, n):
                        if i == 0 and k == n - 1:
                            continue  # wrap-adjacent
                        # Budget checks (cheap and frequent)
                        if evals >= max_evals or moves >= max_moves:
                            break
                        if (time.perf_counter() - start) * 1000.0 >= time_cap_ms:
                            break

                        delta, feasible = atsp_two_opt_delta_and_feasible(t, i, k)
                        evals += 1
                        if feasible and delta < -EPS:
                            apply_two_opt_inplace(t, i, k)
                            moves += 1
                            improved = True
                            break
                    if improved:
                        break
                    # also break outer i-loop if caps hit
                    if evals >= max_evals or moves >= max_moves or (time.perf_counter() - start) * 1000.0 >= time_cap_ms:
                        break

            improved_offspring.append(t)

        return improved_offspring
    
    # ---- Elimination ----
    def elimination(self, joined_population: list[np.ndarray], joined_fitnesses: np.ndarray, distanceMatrix: np.ndarray) -> Tuple:
        """ 
            (μ+λ) Elimination with duplicate filtering 
        """
        mu = self.mu
        D = distanceMatrix
        
        order_indices = np.argsort(joined_fitnesses, kind='mergesort')
        sorted_population = [joined_population[i] for i in order_indices]
        sorted_fitnesses = joined_fitnesses[order_indices]

        sorted_population = self._unique_by_tuple(sorted_population) # Remove duplicates to preserve some diversity
        # Recompute fitness for the filtered ordering
        sorted_fitnesses = self._compute_fitness_population(sorted_population, D)

        # Keep top μ
        population = sorted_population[:mu]
        population_fitnesses = sorted_fitnesses[:mu]
        
        return population, population_fitnesses


    # ==========================
    # ---- Main EA Loop  ----
    # ==========================

    def optimize(self, filename: str):
        # Read distance matrix from file
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        
        D = distanceMatrix

        # --- Initialisation ---
        population = self.initialisation(D)

        population_fitnesses = self._compute_fitness_population(population, D)
        best_idx = int(np.argmin(population_fitnesses))
        best_tour = population[best_idx].copy()
        best_fit = float(population_fitnesses[best_idx])
        
        diversity_counts: list[int] = []
        
        lamb = self.lamb

        # ===== Evolutionary loop =====
        yourConvergenceTestsHere = True
        while yourConvergenceTestsHere:
            # ---------- Create offspring ----------
            offspring: list[np.ndarray] = []

            while len(offspring) < lamb:
                # --- Selection --- 
                p1, p2 = self.selection(population, population_fitnesses)
                
                # --- Crossover ---
                child = self.crossover(p1, p2, D)

                # --- Mutation ---
                child = self.mutation(child, D)

                offspring.append(child)
            # --- Local Search ---
            offspring = self.local_search(offspring, D)
            
            # evaluate fitness of the offsprings 
            offspring_fitnesses = self._compute_fitness_population(offspring, D)
            joined_fitnesses = np.concatenate((population_fitnesses, offspring_fitnesses))

            # joining offspring with previous population
            joined_population = population + offspring

            # --- Elimination ---
            population, population_fitnesses = self.elimination(joined_population, joined_fitnesses, D)

            # tracking best
            gen_best_idx = int(np.argmin(population_fitnesses))
            gen_best_fit = float(population_fitnesses[gen_best_idx])
            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_tour = population[gen_best_idx].copy()

            # reporting
            meanObjective = float(np.mean(population_fitnesses))
            bestObjective = float(best_fit)
            bestSolution = best_tour.astype(int)
            
            # track diversity (unique edges)
            diversity_counts.append(self._count_unique_edges_in_population(population, D))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        
        # Plot and save diversity evolution
        self._diversity_plot_unique_edges(diversity_counts)

        return 0
    
    # ==========================
    # Initialisation helpers   
    # ==========================
    def _randomised_greedy_initialisation_feasible(self, D: np.ndarray, M: np.ndarray, max_restarts: int = 200) -> np.ndarray:
        """
            1. Precompute nearest neighbors for every city among feasible targets, sorted by distance for speed.
            2. Try up to max_restarts; in each attempt:
                Pick a random start city.
                Build the tour greedily: at each step, choose among the nearest feasible unvisited neighbors, but add a bit of randomness by sampling among the top-k (with k ≤ 5) 
                If the greedy sequence finishes, ensure the cycle closes (last → first is finite). If not, try a simple swap fix near the end.
            3. If all restarts fail, raise an error.
        """
        # feasible neighbors lists sorted by distance 
        n = D.shape[0]
        neighbors = []
        for i in range(n):
            js = np.where(M[i])[0]
            if js.size:
                order = np.argsort(D[i, js], kind='mergesort')
                neighbors.append(js[order])
            else:
                neighbors.append(np.array([], dtype=int))

        for _ in range(max_restarts):
            used = np.zeros(n, dtype=bool)
            tour = np.empty(n, dtype=int)

            start = self.rng.integers(0, n)
            tour[0] = start
            used[start] = True

            feasible = True
            for pos in range(1, n):
                curr = tour[pos - 1]
                cand = neighbors[curr][~used[neighbors[curr]]]
                if cand.size == 0:
                    feasible = False
                    break
                # randomized choice among up to top 5 nearest to promote diversity
                k = min(5, cand.size)
                next_city = self.rng.choice(cand[:k])
                tour[pos] = next_city
                used[next_city] = True

            if feasible:
                # Close the cycle: last -> first must be finite
                if np.isfinite(D[tour[-1], tour[0]]):
                    return tour
                # Try a simple end-fix: swap last with a position that fixes closure
                for swap_pos in range(1, n - 1):
                    tmp = tour.copy()
                    tmp[-1], tmp[swap_pos] = tmp[swap_pos], tmp[-1]
                    if np.isfinite(D[tmp[-1], tmp[0]]):
                        # also check the edge into the swapped position remained finite
                        if np.isfinite(D[tmp[swap_pos - 1], tmp[swap_pos]]) and np.isfinite(D[tmp[swap_pos], tmp[swap_pos + 1]]):
                            return tmp
            # else restart
        raise ValueError("Failed to construct a feasible tour with randomized-greedy.")
    
    def _randomised_initialisation_feasible(self, D: np.ndarray, M: np.ndarray, max_restarts: int = 500) -> np.ndarray:
        """
            Construct a feasible tour using a random walk constrained by feasibility (M):
                - pick a random start,
                - repeatedly choose a random feasible unvisited next city (uniform choice),
                - if stuck, restart,
                - ensure last->first is finite; try a simple end-fix via swapping if needed.
            Returns a permutation that induces only finite edges (including last->first), or raises ValueError.
        """
        n = D.shape[0]

        # Precompute feasible neighbors (unsorted; random will sample uniformly)
        neighbors = []
        for i in range(n):
            js = np.where(M[i])[0]
            neighbors.append(js if js.size else np.array([], dtype=int))

        for _ in range(max_restarts):
            used = np.zeros(n, dtype=bool)
            tour = np.empty(n, dtype=int)

            start = self.rng.integers(0, n)
            tour[0] = start
            used[start] = True

            feasible = True
            for pos in range(1, n):
                curr = tour[pos - 1]
                cand = neighbors[curr][~used[neighbors[curr]]]
                if cand.size == 0:
                    feasible = False
                    break
                # Uniform random among all feasible unvisited neighbors
                next_city = self.rng.choice(cand)
                tour[pos] = next_city
                used[next_city] = True

            if feasible:
                # Close the cycle: last -> first must be finite
                if np.isfinite(D[tour[-1], tour[0]]):
                    return tour
                # Try a simple end-fix: swap last with a position that fixes closure
                for swap_pos in range(1, n - 1):
                    tmp = tour.copy()
                    tmp[-1], tmp[swap_pos] = tmp[swap_pos], tmp[-1]
                    if np.isfinite(D[tmp[-1], tmp[0]]):
                        # also check edges around swapped position remained finite
                        if (np.isfinite(D[tmp[swap_pos - 1], tmp[swap_pos]])
                            and np.isfinite(D[tmp[swap_pos], tmp[swap_pos + 1]])):
                            return tmp
            # else restart and try again
        raise ValueError("Failed to construct a random feasible tour.")
    
    # ==========================
    #  Mutation operators   
    # ==========================
    def _mutate_swap(self, tour: np.ndarray) -> np.ndarray:
        """ 
            Pick two indices at random and swap them
            Return the resulting tour
        """
        i, j = self.rng.choice(tour.size, size=2, replace=False)
        child = tour.copy()
        child[i], child[j] = child[j], child[i]
        return child

    def _mutate_invert(self, tour: np.ndarray) -> np.ndarray:
        """ 
            Pick two indices at random and invert the path from i to j
            Return the resulting tour 
        """
        i, j = self.rng.choice(tour.size, size=2, replace=False)
        if i > j:
            i, j = j, i
        child = tour.copy()
        child[i:j+1] = child[i:j+1][::-1]
        return child

    def _mutate_feasible(self, parent: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
            Try up to `feasible_retry` times to produce a mutated child with finite length
            If unsuccessful, return the parent
        """
        ops = (self._mutate_swap, self._mutate_invert)
        for _ in range(self.feasible_retry):
            op = self.rng.choice(ops)
            child = op(parent)
            if np.isfinite(self._tour_length(child, D)):
                return child
        return parent
    
    # ==========================
    #  Crossover operators   
    # ==========================
    def _recombine_feasible(self, p1: np.ndarray, p2: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
            OX crossover with a light repair attempt: if infeasible, try a few random re-shuffles
            around problematic edges; otherwise fall back to the fitter parent.
        """
        #ops = (self._crossover_OX, self._crossover_ERX)
        #op = self.rng.choice(ops)
        op = self._crossover_OX
        child = op(p1, p2)
        if np.isfinite(self._tour_length(child, D)):
            return child

        # Small local shuffle attempts near offending edges
        n = child.size
        for _ in range(self.feasible_retry):
            # Randomly pick a short segment and shuffle it
            i = self.rng.integers(0, n)
            seg_len = int(max(2, min(10, n // 20)))
            j = (i + seg_len) % n
            if i < j:
                seg = child[i:j].copy()
                self.rng.shuffle(seg)
                child[i:j] = seg
            else:
                seg = np.concatenate((child[i:], child[:j])).copy()
                self.rng.shuffle(seg)
                child[i:] = seg[:n - i]
                child[:j] = seg[n - i:]
            if np.isfinite(self._tour_length(child, D)):
                return child

        # Fallback: return the better of the two parents
        return p1  # caller should pass parents in (best, other) order
    
    def _crossover_OX(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
            Order Crossover (OX): preserves a slice from p1 and the relative order of the remaining
            cities from p2. Produces a permutation (not guaranteed feasible in directed sense).
        """
        n = p1.size
        a, b = sorted(self.rng.choice(n, size=2, replace=False))
        child = -np.ones(n, dtype=int)

        # Copy slice from p1
        child[a:b+1] = p1[a:b+1]
        used = np.zeros(n, dtype=bool)
        used[child[a:b+1]] = True

        # Fill remaining positions with p2 order
        pos = (b + 1) % n
        for city in p2:
            if not used[city]:
                child[pos] = city
                used[city] = True
                pos = (pos + 1) % n
        return child

    def _crossover_ERX(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
            Edge Recombination Crossover (ERX / Edge Crossover).
            Constructs an adjacency (edge) table from both parents and builds a child by
            repeatedly choosing the next city based on:
            1) Preference for neighbors that are edges in BOTH parents (shared edges appear twice),
            2) Otherwise choosing a neighbor whose current adjacency list is smallest,
            3) Ties broken at random,
            4) If no neighbors remain, pick a random unused city.
            Args:
                p1: Parent 1 as a permutation (np.ndarray of ints 0..n-1).
                p2: Parent 2 as a permutation (np.ndarray of ints 0..n-1).
            Returns:
                child: A valid permutation (np.ndarray of dtype=int).
        """
        n = p1.size
        child = -np.ones(n, dtype=int)
        used = np.zeros(n, dtype=bool)
        
        # Build successor-only adjacency (duplicate if shared)
        succ_edges: Dict[int, List[int]] = {int(c): [] for c in p1}
        def add_parent_successors(parent: np.ndarray):
            for i, c in enumerate(parent):
                c = int(c)
                right = int(parent[(i + 1) % n])   # successor only
                succ_edges[c].append(right)

        add_parent_successors(p1)
        add_parent_successors(p2)

        # Helper: remove a city from all adjacency lists (so it can't be chosen again)
        def remove_city_from_all(x: int) -> None:
            for k in succ_edges.keys():
                succ_edges[k] = [u for u in succ_edges[k] if u != x]

        # Start from one of the parents' first cities
        start_candidates = np.array([int(p1[0]), int(p2[0])], dtype=int)
        current = int(self.rng.choice(start_candidates))
        child[0] = current
        used[current] = True
        remove_city_from_all(current)

        # Build the rest of the tour 
        for pos in range(1, n):
            neigh = succ_edges[current]  # neighbors list (may contain duplicates)
            # Normally, used nodes have already been removed from all lists

            if len(neigh) > 0:
                # Prefer neighbors that appear in BOTH parents (duplicates in 'neigh')
                cnt = Counter(neigh)
                shared = [v for v in cnt if cnt[v] > 1] # shared successors get priority

                pool = shared if shared else list(set(neigh))
                # Choose those with the smallest current adjacency size
                sizes = [len(succ_edges[v]) for v in pool]
                min_size = min(sizes)
                best = [v for v, s in zip(pool, sizes) if s == min_size]
                next_city = int(self.rng.choice(np.array(best, dtype=int)))
            else:
                # No neighbors left: pick any random unused city
                remaining = np.where(~used)[0]
                next_city = int(self.rng.choice(remaining))

            child[pos] = next_city
            used[next_city] = True
            current = next_city
            remove_city_from_all(current)

        return child

    # ===========================
    # ---- Selection helpers ----
    # ===========================
    def _tournament_select_idx(self, fitness: np.ndarray, k: int) -> int:
        """
            Return index of the tournament winner (lowest fitness)
        """
        n = fitness.size
        cand = self.rng.integers(0, n, size=k)
        winner = cand[np.argmin(fitness[cand])]
        return int(winner)
    
    @staticmethod
    def _compute_fitness_population(population: list[np.ndarray], D: np.ndarray) -> np.ndarray:
        """ 
            Compute objective function for each tour 
            Returns an array of fitnesses 
        """
        return np.array([r1072969._tour_length(t, D) for t in population], dtype=float)
    
    # ===========================
    # ---- Diversity helpers ----
    # ===========================
    @staticmethod
    def _diversity_plot_unique_edges(diversity_counts) -> None:
        """
            Plot and save image of the number of unique directed edges in the population over generations.
        """
        if len(diversity_counts) > 0:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x = np.arange(1, len(diversity_counts) + 1)
            ax.plot(x, diversity_counts, lw=2, color='tab:blue')
            ax.set_xlabel("Generation")
            ax.set_ylabel("Unique directed edges in population")
            ax.set_title("Population diversity (unique edges) over generations")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("diversity_evolution.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # ===========================
    # ---- Utility functions ----
    # ===========================
    @staticmethod
    def _unique_by_tuple(pop: list[np.ndarray]) -> list[np.ndarray]:
        """
            Remove exact duplicates (tuple hashing). Preserves order.
        """
        seen = set()
        uniq = []
        for t in pop:
            key = tuple(map(int, t))
            if key not in seen:
                seen.add(key)
                uniq.append(t)
        return uniq

    @staticmethod
    def _tour_length(tour: np.ndarray, D: np.ndarray) -> float:
        """
            Compute directed cycle length. Returns np.inf if any edge is infeasible (∞).
        """
        idx = tour
        nxt = np.roll(idx, -1)
        edges = D[idx, nxt]
        # If any edge is non-finite, tour is infeasible
        if np.isfinite(edges).all():
            return float(edges.sum())
        return float(np.inf)

    @staticmethod
    def _finite_outgoing_mask(D: np.ndarray) -> np.ndarray:
        """
            Boolean mask M where M[i, j] is True iff edge i->j is finite and j != i.
        """
        M = np.isfinite(D)
        np.fill_diagonal(M, False)
        return M
    
    @staticmethod
    def _count_unique_edges_in_population(pop: list[np.ndarray], Dmat: np.ndarray) -> int:
        """
            Count the number of unique finite directed edges i->j present across the population.
            Directed matters (i->j is different from j->i).
        """
        uniq: set[tuple[int, int]] = set()
        for t in pop:
            idx = t
            nxt = np.roll(idx, -1)  # last -> first closes the cycle
            # Only count edges with finite cost
            finite_mask = np.isfinite(Dmat[idx, nxt])
            # Add directed pairs for finite edges only
            for i, j, f in zip(idx, nxt, finite_mask):
                if f:
                    uniq.add((int(i), int(j)))
        return len(uniq)


if __name__ == '__main__':
    # You can do quick local tests here (won't run in the grading harness).
    # Example:
    
    #process = psutil.Process(os.getpid())

    #start_time = time.perf_counter()

    a = r1072969(mu=100, lamb=100, k_tournament=7, mutation_rate=0.8, crossover_rate=0.6000000000000001)

    a.optimize("./tour250.csv")
    
    #end_time = time.perf_counter()

    #rss_mb = process.memory_info().rss / (1024**2)  # MB
    #print(f"Elapsed time: {end_time - start_time:.2f}s")
    #print(f"Memory used (RSS): {rss_mb:.2f} MB")

    pass