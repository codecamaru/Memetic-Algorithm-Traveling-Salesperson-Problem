# Minimal evolutionary algorithm for directed TSP with possible ∞ (no-edge) distances.
import Reporter
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, List, Optional, Hashable, Dict
from collections import Counter
import random

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
        D = distanceMatrix
        n = D.shape[0]
        M = self._finite_outgoing_mask(D)

        mu = self.mu

        # ---- Initialize a feasible population ----
        # Carol: vamos a probar a aplicar el smart initialisation solo a una parte de la población inicial, y lo otro completamente aleatorio pero feasible. Luego puedes probar con completamente aleatorio instead.
        
        population: list[np.ndarray] = []

        # Compute counts
        num_greedy = max(1, int(round(0.20 * mu)))        # ensure at least 1 greedy
    
        greedy_tries, random_tries = 0, 0
        max_total_tries = 10_000
        
        # First: inject high-quality greedy tours (~20%)
        while len(population) < num_greedy and (greedy_tries + random_tries) < max_total_tries:
            greedy_tries += 1 
            try:
                tour = self._randomized_greedy_initialisation_feasible(D, M)
                population.append(tour)
            except ValueError:
                # Highly constrained instance; keep trying
                continue

        # Second: fill remaining with random feasible (~80%)
        while len(population) < mu and (greedy_tries + random_tries) < max_total_tries:
            random_tries += 1
            try:
                tour = self._random_initialisation_feasible(D, M)
                population.append(tour)
            except ValueError:
                # If random feasible fails, keep trying; may be highly constrained
                continue

        # Fallback: as a last resort, fill the rest with pure random permutations (may be infeasible)
        # EA will attempt to repair via variation; this should rarely happen.
        while len(population) < mu:
            population.append(self.rng.permutation(n))
        
        return population
            
    # ---- Selection ----
    def selection(self, population: list[np.ndarray], fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def crossover(self, p1: np.ndarray, p2: np.ndarray, D: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.crossover_rate:
            child = self._recombine_feasible(p1, p2, D)
        else:
            child = p1.copy()

        return child

    # ---- Mutation ----
    def mutation(self, child: np.ndarray, D: np.ndarray) -> np.ndarray:
        # Mutation with probability
        if self.rng.random() < self.mutation_rate:
            child = self._mutate_feasible(child, D)
        return child
    
    # ---- Elimination ----
    def elimination(self, joined_population: list[np.ndarray], joined_fitnesses: np.ndarray, distanceMatrix: np.ndarray) -> Tuple:
        """ (μ+λ) Elitist survival with duplicate filtering """
        mu = self.mu
        D = distanceMatrix
        # Sort by fitness (ascending), keep feasible first (np.inf automatically sinks to end)
        order = np.argsort(joined_fitnesses, kind='mergesort')
        sorted_pop = [joined_population[i] for i in order]
        sorted_fit = joined_fitnesses[order]

        # Remove duplicates to preserve some diversity
        sorted_pop = self._unique_by_tuple(sorted_pop)
        # Recompute fitness for the filtered ordering
        # (We can align by recomputing for safety; but reuse the sorted order's fitness where possible)
        # Simpler: recompute
        sorted_fit = self._compute_fitness_population(sorted_pop, D)

        # Keep top μ
        population = sorted_pop[:mu]
        population_fitnesses = sorted_fit[:mu]
        
        return population, population_fitnesses


    # ==========================
    # ---- Main optimize()  ----
    # ==========================

    def optimize(self, filename: str):
        # ---- Read distance matrix exactly as in template (do not change) ----
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

            # Recombination + Mutation to produce λ children
            while len(offspring) < lamb:
                # --- Selection --- (k-tournament)
                p1, p2 = self.selection(population, population_fitnesses)
                
                # --- Crossover ---
                child = self.crossover(p1, p2, D)

                # --- Mutation ---
                child = self.mutation(child, D)

                offspring.append(child)

            # evaluate fitness of the offsprings 
            offspring_fitnesses = self._compute_fitness_population(offspring, D)
            joined_fitnesses = np.concatenate((population_fitnesses, offspring_fitnesses))

            # joining offspring with previous population
            joined_population = population + offspring

            # ---------- Elimination ----------
            population, population_fitnesses = self.elimination(joined_population, joined_fitnesses, D)

            # ---------- Track best ----------
            gen_best_idx = int(np.argmin(population_fitnesses))
            gen_best_fit = float(population_fitnesses[gen_best_idx])
            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_tour = population[gen_best_idx].copy()

            # ---------- Reporting ----------
            # Mean objective: use numeric mean, with infeasible counted as +∞; for CSV readability,
            # convert to float (will print 'inf' if infeasible tours remain).
            meanObjective = float(np.mean(population_fitnesses))
            bestObjective = float(best_fit)
            bestSolution = best_tour.astype(int)
            
            # ---------- Diversity (unique directed edges) ----------
            diversity_counts.append(self._count_unique_edges_in_population(population, D))

            # Leave the next three lines as they are (per assignment)
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        
        # Plot and save diversity evolution
        self._diversity_plot_unique_edges(diversity_counts)

        return 0
    
    #  Initialisation helpers 
    def _randomized_greedy_initialisation_feasible(self, D: np.ndarray, M: np.ndarray, max_restarts: int = 200) -> np.ndarray:
        """
        1. Precompute nearest neighbors for every city among feasible targets, sorted by distance for speed.
        2.Try up to max_restarts; in each attempt:
            Pick a random start city.
            Build the tour greedily: at each step, choose among the nearest feasible unvisited neighbors, but add a bit of randomness by sampling among the top-k (with k ≤ 5) to improve diversity and avoid traps.
            If the greedy sequence finishes, ensure the cycle closes (last → first is finite). If not, try a simple swap fix near the end.
        3. If all restarts fail, raise an error.
        
        Construct a *feasible* tour using a randomized greedy heuristic:
        - pick a random start,
        - repeatedly choose the nearest feasible next city with a small randomization (biased choice),
        - if stuck, restart.
        Returns a permutation that induces only finite edges (including last->first), or raises ValueError.
        """
        n = D.shape[0]
        # Precompute neighbor lists sorted by distance for speed.
        # neighbors[i] = array of target cities j (j!=i) with finite D[i,j], sorted by D[i,j]
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
                # Biased randomized choice among up to top 5 nearest to promote diversity
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
    
    def _random_initialisation_feasible(self, D: np.ndarray, M: np.ndarray, max_restarts: int = 500) -> np.ndarray:
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
    # ---- Mutation operators   -----
    # ==========================

    def _mutate_swap(self, tour: np.ndarray) -> np.ndarray:
        i, j = self.rng.choice(tour.size, size=2, replace=False)
        child = tour.copy()
        child[i], child[j] = child[j], child[i]
        return child

    def _mutate_insert(self, tour: np.ndarray) -> np.ndarray:
        i, j = self.rng.choice(tour.size, size=2, replace=False)
        if i > j:
            i, j = j, i
        child = tour.copy()
        city = child[j]
        child = np.delete(child, j)
        child = np.insert(child, i, city)
        return child

    def _mutate_invert(self, tour: np.ndarray) -> np.ndarray:
        i, j = self.rng.choice(tour.size, size=2, replace=False)
        if i > j:
            i, j = j, i
        child = tour.copy()
        child[i:j+1] = child[i:j+1][::-1]
        return child

    def _mutate_feasible(self, parent: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Try up to `feasible_retry` times to produce a mutated child with finite length.
        If unsuccessful, return the parent (conservative fallback).
        """
        ops = (self._mutate_swap, self._mutate_insert, self._mutate_invert)
        for _ in range(self.feasible_retry):
            op = self.rng.choice(ops)
            child = op(parent)
            if np.isfinite(self._tour_length(child, D)):
                return child
        return parent
    
    # ==========================
    # ---- Crossover operators   -----
    # ==========================

    def _recombine_feasible(self, p1: np.ndarray, p2: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        OX crossover with a light repair attempt: if infeasible, try a few random re-shuffles
        around problematic edges; otherwise fall back to the fitter parent.
        """
        child = self._crossover_OX(p1, p2) # habrá que cambiar esto maybe para saber cuál coger
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

    def _crossover_CX(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Cycle Crossover (CX): decomposes positions into cycles by mapping p2's values
        back to their indices in p1. Alternates cycles between parents to produce
        a valid permutation. Not guaranteed feasible in directed sense.

        Args:
            p1: Parent 1 as a permutation (np.ndarray of ints 0..n-1).
            p2: Parent 2 as a permutation (np.ndarray of ints 0..n-1).

        Returns:
            child: A valid permutation (np.ndarray of dtype=int).
        """
        n = p1.size
        child = -np.ones(n, dtype=int)

        # Map each value in p1 -> its position index (requires values in 0..n-1)
        pos_in_p1 = np.empty(n, dtype=int)
        for i, v in enumerate(p1):
            pos_in_p1[v] = i

        visited = np.zeros(n, dtype=bool)
        cycles = []

        # --- Build cycles of positions ---
        # Starting from each unvisited position, follow the chain:
        # i -> p2[i] -> index in p1 where that value occurs -> repeat until we return.
        for start in range(n):
            if visited[start]:
                continue
            cycle = []
            i = start
            while not visited[i]:
                visited[i] = True
                cycle.append(i)
                next_val = p2[i]
                i = pos_in_p1[next_val]
            cycles.append(np.array(cycle, dtype=int))

        # Randomly decide which parent supplies the first cycle (to avoid bias)
        start_with_p1 = bool(self.rng.choice(2))

        use_p1 = start_with_p1
        for cyc in cycles:
            if use_p1:
                child[cyc] = p1[cyc]
            else:
                child[cyc] = p2[cyc]
            use_p1 = not use_p1  # alternate parent for next cycle

        return child

    
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

        # --- Build adjacency list with duplicates (shared edges appear twice) ---
        # edges[c] is a list (not a set) to preserve duplicate neighbors
        edges: Dict[int, List[int]] = {int(c): [] for c in p1}

        def add_parent_edges(parent: np.ndarray):
            for i, c in enumerate(parent):
                c = int(c)
                left = int(parent[(i - 1) % n])
                right = int(parent[(i + 1) % n])
                edges[c].extend([left, right])

        add_parent_edges(p1)
        add_parent_edges(p2)

        # --- Helper: remove a city from all adjacency lists (so it can't be chosen again) ---
        def remove_city_from_all(x: int) -> None:
            for k in edges.keys():
                lst = edges[k]
                if lst:
                    # remove all occurrences of x (handles duplicates cleanly)
                    edges[k] = [u for u in lst if u != x]

        # --- Choose a starting city: randomly between the two parents' first cities ---
        start_candidates = np.array([int(p1[0]), int(p2[0])], dtype=int)
        current = int(self.rng.choice(start_candidates))
        child[0] = current
        used[current] = True
        remove_city_from_all(current)

        # --- Build the rest of the tour ---
        for pos in range(1, n):
            neigh = edges[current]  # neighbors list (may contain duplicates)
            # Normally, used nodes have already been removed from all lists
            candidates = neigh  # still keep as list for frequency/dup detection

            if len(candidates) > 0:
                # Prefer neighbors that appear in BOTH parents (duplicates in 'neigh')
                cnt = Counter(neigh)
                shared = [v for v in cnt if cnt[v] > 1]

                pool = shared if shared else list(set(candidates))  # ensure uniqueness beyond duplicates
                # Choose those with the smallest current adjacency size
                sizes = [len(edges[v]) for v in pool]
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

    def _crossover_PMX(self,parent1: Sequence[Hashable],parent2: Sequence[Hashable],cut_points: Optional[Tuple[int, int]] = None,) -> Tuple[List[Hashable], List[Hashable]]:
        """
        Perform PMX (Partially Mapped Crossover) on two permutation parents.
        Args:
            parent1: First parent (permutation of unique, hashable genes).
            parent2: Second parent (same genes in some order).
            cut_points: Optional (c1, c2). If None, chosen randomly.
                        Inclusive indices: 0 <= c1 < c2 < n
        Returns:
            (child1, child2): Two offspring lists.
        Raises:
            ValueError: If parents have different lengths, are not permutations,
                        or cut_points are invalid.
        """
        n = len(parent1)
        if len(parent2) != n:
            raise ValueError("Parents must have the same length.")
        if n < 2:
            # Not enough length to crossover—return copies
            return list(parent1), list(parent2)

        # Validate permutations and gene sets
        if len(set(parent1)) != n or len(set(parent2)) != n:
            raise ValueError("PMX requires parents to be permutations (all genes unique).")
        if set(parent1) != set(parent2):
            raise ValueError("Parents must contain the exact same set of genes.")

        # Choose/validate crossover points
        if cut_points is None:
            c1, c2 = sorted(self.rng.sample(range(n), 2))
        else:
            c1, c2 = cut_points
            if not (0 <= c1 < c2 < n):
                raise ValueError("cut_points must satisfy 0 <= c1 < c2 < n.")

        return self._pmx_pair(parent1, parent2, c1, c2)

    def _pmx_pair(self, p1: Sequence[Hashable], p2: Sequence[Hashable], c1: int, c2: int) -> Tuple[List[Hashable], List[Hashable]]:
        """
        Build both PMX offspring using inclusive indices [c1..c2].
        """
        n = len(p1)
        child1 = [None] * n
        child2 = [None] * n

        # 1) Copy the crossover segments directly
        child1[c1 : c2 + 1] = p1[c1 : c2 + 1]
        child2[c1 : c2 + 1] = p2[c1 : c2 + 1]

        # 2) Build mapping dictionaries for conflict resolution chains
        # For child1, we import from p2 and map through p2->p1 on the segment.
        map_p2_to_p1 = {p2[i]: p1[i] for i in range(c1, c2 + 1)}
        # For child2, we import from p1 and map through p1->p2 on the segment.
        map_p1_to_p2 = {p1[i]: p2[i] for i in range(c1, c2 + 1)}

        def fill_outside_segment(child: List[Hashable], donor: Sequence[Hashable], mapping: dict):
            """
            Fill positions outside [c1..c2] from donor, resolving duplicates
            by following the mapping chain until an unused gene is found.
            """
            segment = set(child[c1 : c2 + 1])  # Faster membership checks
            for i in range(n):
                if c1 <= i <= c2:
                    continue
                gene = donor[i]
                # Follow conflict chain if gene already in the copied segment
                while gene in segment:
                    gene = mapping[gene]
                child[i] = gene

        fill_outside_segment(child1, p2, map_p2_to_p1)
        fill_outside_segment(child2, p1, map_p1_to_p2)

        return child1, child2
    
    # ===========================
    # ---- Selection helpers ----
    # ===========================
    
    def _tournament_select_idx(self, fitness: np.ndarray, k: int) -> int:
        """Return index of the tournament winner (lowest fitness)."""
        n = fitness.size
        cand = self.rng.integers(0, n, size=k)
        winner = cand[np.argmin(fitness[cand])]
        return int(winner)
    
    @staticmethod
    def _compute_fitness_population(population: list[np.ndarray], D: np.ndarray) -> np.ndarray:
        """ Compute objective function for each tour 
            Returns an array of fitnesses """
        return np.array([r1072969._tour_length(t, D) for t in population], dtype=float)
    
    # ===========================
    # ---- Diversity helpers ----
    # ===========================
    @staticmethod
    def _diversity_plot_unique_edges(diversity_counts) -> None:
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
        """Remove exact duplicates (tuple hashing). Preserves order."""
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
        tour: 1D array of unique city indices, cover all cities exactly once.
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
        Used to speed up feasible construction/repair.
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
            # Only count edges with finite cost (feasible directed arcs)
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

    a = r1072969(mu=200, lamb=270, k_tournament=7, mutation_rate=0.8, crossover_rate=0.6000000000000001)

    a.optimize("./tour250.csv")
    
    #end_time = time.perf_counter()

    #rss_mb = process.memory_info().rss / (1024**2)  # MB
    #print(f"Elapsed time: {end_time - start_time:.2f}s")
    #print(f"Memory used (RSS): {rss_mb:.2f} MB")

    pass