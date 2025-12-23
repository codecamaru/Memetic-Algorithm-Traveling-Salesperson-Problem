# Minimal evolutionary algorithm for directed TSP with possible ∞ (no-edge) distances.
# Conforms to the KU Leuven template and Reporter interface.
# Heavily annotated to serve as a solid base for your individual phase.

import Reporter
import numpy as np

#import psutil
#import os
#import time


class r1072969:
    def __init__(self,
                 rng_seed: int | None = None,
                 mu: int | None = None,            # population size (optional: auto if None)
                 lamb: int | None = None,          # offspring per generation (optional: auto if None)
                 k_tournament: int | None = None,  # tournament size (optional: auto if None)
                 mutation_rate: float = 0.9,       # probability to mutate a selected parent
                 crossover_rate: float = 0.9,      # probability to recombine two parents
                 feasible_retry: int = 20          # retries to obtain a feasible variation
                 ):
        """
        Parameters are modest defaults aimed at n in [100..1000] with ~5 min CPU budget.
        You can tune these later (e.g., for your experiments in session 3).
        """
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.rng = np.random.default_rng(rng_seed)

        # Hyperparameters (may be auto-set after reading instance size)
        self._mu_in = mu
        self._lamb_in = lamb
        self._k_in = k_tournament
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.feasible_retry = feasible_retry

    # ===========================
    # ---- Utility functions ----
    # ===========================

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
    def _evaluate_population(pop: list[np.ndarray], D: np.ndarray) -> np.ndarray:
        """Vector-ish evaluation for a list of tours -> array of fitnesses (lower is better)."""
        # Keep it simple; loop is fine because evaluate is O(n) anyway.
        return np.array([r1072969._tour_length(t, D) for t in pop], dtype=float)

    @staticmethod
    def _finite_outgoing_mask(D: np.ndarray) -> np.ndarray:
        """
        Boolean mask M where M[i, j] is True iff edge i->j is finite and j != i.
        Used to speed up feasible construction/repair.
        """
        M = np.isfinite(D)
        np.fill_diagonal(M, False)
        return M

    # ===============================
    # ---- Initialization (feas) ----
    # ===============================

    def _randomized_greedy_feasible(self, D: np.ndarray, M: np.ndarray, max_restarts: int = 200) -> np.ndarray:
        """
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

    # ==========================
    # ---- Selection (k-t) -----
    # ==========================

    def _tournament_select_idx(self, fitness: np.ndarray, k: int) -> int:
        """Return index of the tournament winner (lowest fitness)."""
        n = fitness.size
        cand = self.rng.integers(0, n, size=k)
        winner = cand[np.argmin(fitness[cand])]
        return int(winner)

    # ==========================
    # ---- Variation ops   -----
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

    def _recombine_feasible(self, p1: np.ndarray, p2: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        OX crossover with a light repair attempt: if infeasible, try a few random re-shuffles
        around problematic edges; otherwise fall back to the fitter parent.
        """
        child = self._crossover_OX(p1, p2)
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
    # ==========================
    # ---- Elimination (μ+λ) ----
    # ==========================

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

    # ==========================
    # ---- Main optimize()  ----
    # ==========================

    def optimize(self, filename: str):
        # ---- Read distance matrix exactly as in template (do not change) ----
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        D = distanceMatrix
        n = D.shape[0]
        M = self._finite_outgoing_mask(D)

        # Auto-parameterization (safe defaults for 100..1000)
        mu = self._mu_in if self._mu_in is not None else int(np.clip(n // 10, 20, 80))
        lamb = self._lamb_in if self._lamb_in is not None else mu
        k = self._k_in if self._k_in is not None else int(np.clip(7, 2, mu))

        # ---- Initialize a feasible population ----
        population: list[np.ndarray] = []
        tries, target = 0, mu
        while len(population) < target and tries < 10_000:
            tries += 1
            try:
                tour = self._randomized_greedy_feasible(D, M)
                population.append(tour)
            except ValueError:
                # Highly constrained instance; keep trying
                continue
        if len(population) == 0:
            # As a last resort, fall back to random permutations (may be infeasible),
            # but EA will attempt to repair via variation; this should rarely happen.
            for _ in range(mu):
                population.append(self.rng.permutation(n))

        fitness = self._evaluate_population(population, D)
        best_idx = int(np.argmin(fitness))
        best_tour = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        # ===== Evolutionary loop =====
        yourConvergenceTestsHere = True
        while yourConvergenceTestsHere:
            # ---------- Create offspring ----------
            offspring: list[np.ndarray] = []

            # Recombination + Mutation to produce λ children
            while len(offspring) < lamb:
                # Parent selection (k-tournament)
                p1_idx = self._tournament_select_idx(fitness, k)
                p2_idx = self._tournament_select_idx(fitness, k)
                while p2_idx == p1_idx and mu > 1:
                    p2_idx = self._tournament_select_idx(fitness, k)
                p1, p2 = population[p1_idx], population[p2_idx]

                # Ensure p1 is the better parent (helps in fallback decisions)
                if fitness[p2_idx] < fitness[p1_idx]:
                    p1, p2 = p2, p1

                # Crossover with probability
                if self.rng.random() < self.crossover_rate:
                    child = self._recombine_feasible(p1, p2, D)
                else:
                    child = p1.copy()

                # Mutation with probability
                if self.rng.random() < self.mutation_rate:
                    child = self._mutate_feasible(child, D)

                offspring.append(child)

            # ---------- Evaluate offspring ----------
            off_fit = self._evaluate_population(offspring, D)

            # ---------- (μ+λ) Elitist survival with duplicate filtering ----------
            combined = population + offspring
            combined_fit = np.concatenate((fitness, off_fit))

            # Sort by fitness (ascending), keep feasible first (np.inf automatically sinks to end)
            order = np.argsort(combined_fit, kind='mergesort')
            sorted_pop = [combined[i] for i in order]
            sorted_fit = combined_fit[order]

            # Remove duplicates to preserve some diversity
            sorted_pop = self._unique_by_tuple(sorted_pop)
            # Recompute fitness for the filtered ordering
            # (We can align by recomputing for safety; but reuse the sorted order's fitness where possible)
            # Simpler: recompute
            sorted_fit = self._evaluate_population(sorted_pop, D)

            # Keep top μ
            population = sorted_pop[:mu]
            fitness = sorted_fit[:mu]

            # ---------- Track best ----------
            gen_best_idx = int(np.argmin(fitness))
            gen_best_fit = float(fitness[gen_best_idx])
            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_tour = population[gen_best_idx].copy()

            # ---------- Reporting ----------
            # Mean objective: use numeric mean, with infeasible counted as +∞; for CSV readability,
            # convert to float (will print 'inf' if infeasible tours remain).
            meanObjective = float(np.mean(fitness))
            bestObjective = float(best_fit)
            bestSolution = best_tour.astype(int)

            # Leave the next three lines as they are (per assignment)
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0


if __name__ == '__main__':
    # You can do quick local tests here (won't run in the grading harness).
    # Example:
    
    #process = psutil.Process(os.getpid())

    #start_time = time.perf_counter()

    a = r1072969(mu=20, lamb=270, k_tournament=7, mutation_rate=0.8, crossover_rate=0.6000000000000001)

    a.optimize("./tour50.csv")
    
    #end_time = time.perf_counter()

    #rss_mb = process.memory_info().rss / (1024**2)  # MB
    #print(f"Elapsed time: {end_time - start_time:.2f}s")
    #print(f"Memory used (RSS): {rss_mb:.2f} MB")

    pass