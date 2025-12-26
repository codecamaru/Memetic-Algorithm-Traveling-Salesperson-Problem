# Minimal evolutionary algorithm for directed TSP with possible ∞ (no-edge) distances.
import Reporter
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from collections import Counter
import time
from numba import jit

@jit(nopython=True)
def fast_two_opt_delta(tour, i, k, distanceMatrix, n):
    """
    Compute the change in tour length if we reverse segment (i+1 .. k),
    and check that all new arcs are finite. Works for directed/asymmetric matrices.
    JIT-compiled for speed.
    """
    if (i + 1) % n == k or (k + 1) % n == i:
        return 0.0, False

    a = tour[i]
    b = tour[(i + 1) % n]
    c = tour[k]
    d = tour[(k + 1) % n]

    # New endpoint arcs must be finite
    if not (np.isfinite(distanceMatrix[a, c]) and np.isfinite(distanceMatrix[b, d])):
        return 0.0, False

    delta = -distanceMatrix[a, b] - distanceMatrix[c, d] + distanceMatrix[a, c] + distanceMatrix[b, d]

    # Internal edges loop - fast in Numba
    for t in range(i + 1, k):
        u = tour[t]
        v = tour[t + 1]
        
        # New arc v->u
        val_vu = distanceMatrix[v, u]
        if not np.isfinite(val_vu):
            return 0.0, False
            
        val_uv = distanceMatrix[u, v]
        delta += (val_vu - val_uv)

    return float(delta), True

@jit(nopython=True)
def fast_tour_length(tour, distanceMatrix, n):
    """
    Compute tour length avoiding temporary array allocations.
    """
    obj = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        dist = distanceMatrix[a, b]
        if not np.isfinite(dist):
            return np.inf
        obj += dist
    return float(obj)

@jit(nopython=True)
def fast_crossover_OX(p1, p2, n, a, b):
    """
    Order Crossover (OX) compiled.
    a, b are the start/end indices (inclusive) of the slice from p1.
    """
    child = np.empty(n, dtype=np.int64) # or whatever dtype tour is
    # Fill with -1 or just track via pointer
    
    # We need a boolean mask for used cities
    used = np.zeros(n, dtype=np.bool_)
    
    # 1. Copy slice from p1
    # Note: b is inclusive here to match Python slice a:b+1
    count = 0
    # simple loop to copy and mark used
    if a <= b: # Normal range
        for i in range(a, b + 1):
            val = p1[i]
            child[i] = val
            used[val] = True
    else: 
        # Should not happen with current python logic which sorts a,b, 
        # but handled just in case or assume sorted
        pass 
        
    # 2. Fill remaining from p2
    # Current pos in child to fill
    pos = (b + 1) % n
    
    for i in range(n):
        city = p2[i]
        if not used[city]:
            child[pos] = city
            # used[city] = True # not strictly needed if we purely fill
            pos = (pos + 1) % n
            
    return child

@jit(nopython=True)
def fast_greedy_search_replacement(tour, distanceMatrix):
    """
    Reconstructs a tour started from its lowest-cost edge, extending it 
    by greedily visiting the nearest unvisited neighbor.
    Returns the original tour if reconstruction fails or is infeasible.
    """
    n = len(tour)
    
    # 1. Find the edge in the current tour with the lowest cost
    best_u = -1
    best_v = -1
    min_dist = np.inf
    
    # Iterate through current tour edges
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        d = distanceMatrix[u, v]
        # We look for the minimum edge weight
        if d < min_dist:
            min_dist = d
            best_u = u
            best_v = v
            
    # If the best edge is infinite (infeasible), return parent
    if not np.isfinite(min_dist):
        return tour

    # 2. Initialize new tour with the best edge
    new_tour = np.empty(n, dtype=tour.dtype)
    used = np.zeros(n, dtype=np.bool_)
    
    new_tour[0] = best_u
    new_tour[1] = best_v
    used[best_u] = True
    used[best_v] = True
    
    current_pos = 1
    current_node = best_v
    
    # 3. Greedily fill the rest of the tour
    for _ in range(2, n):
        # Find closest unvisited neighbor to current_node
        best_cand = -1
        best_cand_dist = np.inf
        
        # Linear scan for the nearest unvisited neighbor
        for cand in range(n):
            if not used[cand]:
                d_cand = distanceMatrix[current_node, cand]
                if np.isfinite(d_cand):
                    if d_cand < best_cand_dist:
                        best_cand_dist = d_cand
                        best_cand = cand
        
        if best_cand == -1:
            # No reachable unvisited neighbor found -> failure
            return tour
            
        current_pos += 1
        new_tour[current_pos] = best_cand
        used[best_cand] = True
        current_node = best_cand
        
    # 4. Check if the cycle can be closed
    if not np.isfinite(distanceMatrix[new_tour[-1], new_tour[0]]):
        return tour
        
    return new_tour

@jit(nopython=True)
def fast_select_winner(fitness, candidates):
    """
    Tournament selection winner: index with lowest fitness among candidates.
    """
    best_idx = candidates[0]
    best_fit = fitness[candidates[0]]
    
    for i in range(1, len(candidates)):
        idx = candidates[i]
        fit = fitness[idx]
        if fit < best_fit:
            best_fit = fit
            best_idx = idx
    return best_idx

@jit(nopython=True)
def fast_mutate_swap(tour, i, j):
    """
    Swap elements at indices i and j.
    """
    child = tour.copy()
    val_i = child[i]
    child[i] = child[j]
    child[j] = val_i
    return child

@jit(nopython=True)
def fast_mutate_invert(tour, i, j):
    """
    Invert segment between i and j (inclusive).
    """
    if i > j:
        i, j = j, i
    child = tour.copy()
    # Numba supports standard slice assignment
    child[i : j + 1] = child[i : j + 1][::-1]
    return child

@jit(nopython=True)
def fast_two_opt_search_inner(tour, distanceMatrix, max_evals, max_moves, time_limit_ignored=0):
    """
    Inner loop of 2-opt local search.
    Returns the improved tour.
    """
    t = tour.copy()
    n = len(t)
    EPS = 1e-12
    
    # We cannot use time.perf_counter() in nopython mode easily without objmode, 
    # so we rely on max_evals/max_moves to bound execution.
    evals = 0
    moves = 0
    improved = True
    
    while improved:
        improved = False
        for i in range(n):
            for k in range(i + 2, n):
                if i == 0 and k == n - 1:
                    pass 
                else:
                    delta, feasible = fast_two_opt_delta(t, i, k, distanceMatrix, n)
                    evals += 1
                    
                    if feasible and delta < -EPS:
                        # Apply move
                        t[i + 1 : k + 1] = t[i + 1 : k + 1][::-1]
                        moves += 1
                        improved = True
                        # restart search from first improvement (first improvement strategy)
                        # or break to restart outer loop?
                        # The original code did `break` then `if improved: break` to restart the while loop.
                        break 
                
                # Check limits
                if evals >= max_evals or moves >= max_moves:
                    return t

            if improved:
                # Restart the scanner
                break 
                
            if evals >= max_evals or moves >= max_moves:
                return t
                
    return t

@jit(nopython=True)
def fast_three_opt_search(tour, distanceMatrix, max_evals):
    """
    Apply 3-opt reconnection.
    We iterate over 3 cuts (i, j, k) and try to reconnect segments [start..i], [i+1..j], [j+1..k], [k+1..end]
    in a way that does NOT reverse strict sub-segments (to avoid infinite edges in asymmetric TSP),
    but effectively permutes the ordering of the 3 mutable segments.
    
    Segments:
    A: 0...i
    B: i+1...j
    C: j+1...k
    D: k+1...n-1 (implicit end)
    
    Standard cycle: A -> B -> C -> D -> A
    
    Possible non-reversing reconnects involved moving B or C.
    For example: A -> C -> B -> D -> A
    
    This is equivalent to a 'shift' or 'reinsertion' move in some contexts.
    """
    t = tour.copy()
    n = len(t)
    EPS = 1e-12
    improved = True
    evals = 0
    
    while improved:
        improved = False
        # We need 3 cuts.
        # i from 0 to n-3
        # j from i+1 to n-2
        # k from j+1 to n-1
        
        # Limit search space for performance if needed, but here we try logic
        for i in range(n - 5): # heuristic buffer
            a = t[i]
            b = t[i+1] # outgoing edge from A
            if not np.isfinite(distanceMatrix[a, b]): continue
            
            for j in range(i + 2, n - 3):
                c = t[j]
                d = t[j+1]
                if not np.isfinite(distanceMatrix[c, d]): continue
                
                # We check the move: A->C... -> B... -> D...
                # i.e. insert segment C (from j+1 to k) between A and B
                
                for k in range(j + 2, n - 1):
                    e = t[k]
                    f = t[k+1]
                    if not np.isfinite(distanceMatrix[e, f]): continue
                    
                    evals += 1
                    
                    # Current edges: (a,b), (c,d), (e,f)
                    # Cost removed: D[a,b] + D[c,d] + D[e,f]
                    current_cost = distanceMatrix[a,b] + distanceMatrix[c,d] + distanceMatrix[e,f]
                    
                    # Candidate geometry: A -> (j+1...k) -> (i+1...j) -> (k+1...)
                    # New edges: (a, t[j+1]), (t[k], t[i+1]), (t[j], f)
                    # Let's map these points:
                    # u = t[j+1] (start of segment 3)
                    # v = t[k] (end of segment 3)
                    # w = t[i+1] (start of segment 2)
                    # x = t[j] (end of segment 2)
                    
                    u = t[j+1]
                    v = t[k]
                    w = t[i+1]
                    x = t[j]
                    
                    if np.isfinite(distanceMatrix[a, u]) and np.isfinite(distanceMatrix[v, w]) and np.isfinite(distanceMatrix[x, f]):
                        new_cost = distanceMatrix[a, u] + distanceMatrix[v, w] + distanceMatrix[x, f]
                        if new_cost < current_cost - EPS:
                            # Apply shift: 
                            # Old: [...i] [i+1...j] [j+1...k] [k+1...]
                            # New: [...i] [j+1...k] [i+1...j] [k+1...]
                            
                            # Construct new tour array
                            # 1. 0..i
                            # 2. j+1..k
                            # 3. i+1..j
                            # 4. k+1..n
                            
                            # It's faster to do slice assignment if we are careful
                            # But standard numpy concatenate often easier to reason about for 3 parts
                            
                            seg1 = t[0 : i+1]
                            seg2 = t[i+1 : j+1]
                            seg3 = t[j+1 : k+1]
                            seg4 = t[k+1 : n]
                            
                            t = np.concatenate((seg1, seg3, seg2, seg4))
                            improved = True
                            break
                            
                    if evals > max_evals:
                        return t
                if improved: break
            if improved: break
            
    return t

@jit(nopython=True)
def fast_crossover_ERX(p1, p2, n, rand_vals):
    """
    Edge Recombination Crossover (ERX) optimized with Numba.
    Builds a child preserving adjacency information from parents.
    
    rand_vals: Array of random floats [0, 1) to use for decision making.
    """
    # 1. Build Adjacency Map (Directed)
    # Max degree is 2 (one from p1, one from p2)
    # We use a flat array or fixed size matrix: adj[city, 0] and adj[city, 1]
    # -1 means empty
    adj = np.full((n, 2), -1, dtype=np.int32)
    
    # Fill from P1
    for i in range(n):
        u = p1[i]
        v = p1[(i + 1) % n]
        adj[u, 0] = v
        
    # Fill from P2
    for i in range(n):
        u = p2[i]
        v = p2[(i + 1) % n]
        if adj[u, 0] != v:
            adj[u, 1] = v
    
    # Track used
    used = np.zeros(n, dtype=np.bool_)
    child = np.empty(n, dtype=np.int32)
    
    # Pick start node (from p1[0] usually)
    curr = p1[0]
    child[0] = curr
    used[curr] = True
    
    rand_idx = 0
    max_rand = len(rand_vals)
    
    for i in range(1, n):
        # Determine candidates from adj[curr]
        c1 = adj[curr, 0]
        c2 = adj[curr, 1]
        
        # Scalar candidates
        candA = -1
        candB = -1
        num_cands = 0
        
        if c1 != -1 and not used[c1]:
            candA = c1
            num_cands = 1
        
        if c2 != -1 and c2 != c1 and not used[c2]:
            if num_cands == 0:
                candA = c2
                num_cands = 1
            else:
                candB = c2
                num_cands = 2
            
        best_cand = -1
        
        if num_cands > 0:
            # Pick neighbor with fewest unused neighbors
            
            # Evaluate candA
            degA = 0
            n1 = adj[candA, 0]
            n2 = adj[candA, 1]
            if n1 != -1 and not used[n1]: degA += 1
            if n2 != -1 and n2 != n1 and not used[n2]: degA += 1
            
            min_deg = degA
            best_cand = candA
            tie_count = 1
            
            if num_cands > 1:
                # Evaluate candB
                degB = 0
                n1 = adj[candB, 0]
                n2 = adj[candB, 1]
                if n1 != -1 and not used[n1]: degB += 1
                if n2 != -1 and n2 != n1 and not used[n2]: degB += 1
                
                if degB < min_deg:
                    min_deg = degB
                    best_cand = candB
                    tie_count = 1
                elif degB == min_deg:
                    tie_count = 2
                    
            # Selection from ties
            if tie_count == 2:
                # Random choice between candA and candB
                r = rand_vals[rand_idx % max_rand]
                rand_idx += 1
                if r < 0.5:
                    best_cand = candA
                else:
                    best_cand = candB
                
        else:
            # Dead end: Random jump
            r = rand_vals[rand_idx % max_rand]
            rand_idx += 1
            start_scan = int(r * n)
            
            found = False
            for k in range(n):
                idx = (start_scan + k) % n
                if not used[idx]:
                    best_cand = idx
                    found = True
                    break
            if not found:
                # This should not happen
                best_cand = 0 
                
        child[i] = best_cand
        used[best_cand] = True
        curr = best_cand
        
    return child

class r1072969:
    def __init__(self,
                 rng_seed: int | None = None,
                 mu: int = 100,                         # population size (optional: auto if None) 
                 lamb: int = 100,                       # offspring per generation (optional: auto if None)
                 k_tournament: int = 3,                 # tournament size (optional: auto if None)
                 mutation_rate: float = 0.6,       # probability to mutate a selected parent 
                 crossover_rate: float = 0.9,      # probability to recombine two parents 
                 feasible_retry: int = 20,         # retries to obtain a feasible variation
                 greedy_search_replacement_prob: float = 0.2 # probability to apply greedy search replacement (as percentage of worst population)
                 ):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.rng = np.random.default_rng(rng_seed)

        self.mu = mu
        self.lamb = lamb
        self.k_tournament = k_tournament
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.feasible_retry = feasible_retry
        self.greedy_search_replacement_prob = greedy_search_replacement_prob
        
    # ---- Initialisation ----
    def initialisation(self, distanceMatrix: np.ndarray) -> np.ndarray:
        """ 
            Generate mu individuals (90% greedy, 10% random), all feasible if possible 
        """
        n = distanceMatrix.shape[0]
        M = self._finite_outgoing_mask(distanceMatrix)
        mu = self.mu
   
        population: list[np.ndarray] = []

        num_greedy_individuals = max(1, int(round(0.9 * mu))) # ensure at least 1 greedy
    
        greedy_tries, random_tries = 0, 0
        max_total_tries = 10_000
        
        # greedy tours (~90% population)
        while len(population) < num_greedy_individuals and (greedy_tries + random_tries) < max_total_tries:
            greedy_tries += 1 
            try:
                tour = self._randomised_greedy_initialisation_feasible(distanceMatrix, M)
                population.append(tour)
            except ValueError:
                continue

        # random tours (~10% population)
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
            Applies the local search operator to the offspring.
        """
        if self.rng.random() < 0.3:
            return self.greedy_search_replacement(offspring, distanceMatrix)
        else:
            return self.two_opt_local_search(offspring, distanceMatrix)
    def greedy_search_replacement(self, offspring: list[np.ndarray], distanceMatrix: np.ndarray) -> list[np.ndarray]:
        """ 
            Applies the greedy search replacement operator to the worst 'greedy_search_replacement_proportion' 
            fraction of the offspring.
        """
        greedy_search_replacement_proportion = 0.2
        # Calculate fitness for all offspring
        offspring_fitnesses = self._compute_fitness_population(offspring, distanceMatrix)
        
        # Sort indices: low fitness (good) -> high fitness (bad)
        # We want the worst ones, which are at the end of the sorted array
        sorted_indices = np.argsort(offspring_fitnesses)
        
        n = len(offspring)
        n_worst = int(n * greedy_search_replacement_proportion)
        
        # Identify indices of the worst tours
        # If n_worst is 0, we do nothing
        if n_worst > 0:
            worst_indices = set(sorted_indices[-n_worst:])
        else:
            worst_indices = set()

        new_offspring = []
        for i, child in enumerate(offspring):
            if i in worst_indices:
                new_child = fast_greedy_search_replacement(child, distanceMatrix)
                new_offspring.append(new_child)
            else:
                new_offspring.append(child)
                
        return new_offspring

    def two_opt_local_search(self, offspring: list[np.ndarray], distanceMatrix: np.ndarray) -> list[np.ndarray]:
        """ 
            2-Opt local search using Numba JIT.
        """
        improved_offspring: list[np.ndarray] = []

        # --- Budget ---
        p_ls = 0.6             # apply LS only to 40% of children
        max_evals = 50_000     # increased to allow full search
        max_moves = 2_000       # effectively uncap moves so it finds the local optimum
        time_cap_ms = 50        # increased time cap per individual
        EPS = 1e-12
        TIME_CHECK_FREQ = 64
        
        for child in offspring:
            if self.rng.random() >= p_ls:
                improved_offspring.append(child)
                continue

            t = child.copy()
            if not np.isfinite(self._tour_length(t, distanceMatrix)):
                improved_offspring.append(t)
                continue

            n = t.size
            start_time = time.perf_counter()
            evals = 0
            moves = 0
            
            evals = 0
            moves = 0
            
            # Use JIT-compiled inner loop
            # We ignore time_cap_ms within the JIT function for now, relying on max_evals/max_moves
            t = fast_two_opt_search_inner(t, distanceMatrix, max_evals, max_moves, 0)
                        
            improved_offspring.append(t)

        return improved_offspring
    
    # ---- Elimination ----
    def elimination(self, joined_population: list[np.ndarray], joined_fitnesses: np.ndarray, distanceMatrix: np.ndarray) -> Tuple:
        """ 
            (μ+λ) Elimination with duplicate filtering 
        """
        mu = self.mu
        
        order_indices = np.argsort(joined_fitnesses, kind='mergesort')
        sorted_population = [joined_population[i] for i in order_indices]
        sorted_fitnesses = joined_fitnesses[order_indices]

        sorted_population = self._unique_by_tuple(sorted_population) # Remove duplicates to preserve some diversity
        # Recompute fitness for the filtered ordering
        sorted_fitnesses = self._compute_fitness_population(sorted_population, distanceMatrix)

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
        
        #diversity_counts: list[int] = []
        
        lamb = self.lamb
        
        # ===== Evolutionary loop =====
        self.best_fitness_history = []
        self.stagnation_counter = 0
        STAGNATION_LIMIT = 5
        
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
            
            # --- Diversity promotion ---
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
            
            # Stagnation check
            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_tour = population[gen_best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                
            # Apply 3-opt if stagnated
            if self.stagnation_counter >= STAGNATION_LIMIT:
                print(f"Stagnation detected ({self.stagnation_counter} gens). Applying 3-opt to top individuals.")
                self.stagnation_counter = 0 # reset or keep counting? Reset to give it time to work.
                
                # Apply 3-opt to the best half of population to try and break out
                # or just the top K?
                # Let's do top 10% or at least 5 individuals
                k_dest = max(5, int(0.1 * self.mu))
                
                # Population is already sorted by fitness from elimination step? 
                # elimination returns sorted population usually?
                # "population = sorted_population[:mu]" in elimination. 
                # Yes, index 0 is best.
                
                for i in range(k_dest):
                    # Heavy local search
                    improved_tour = fast_three_opt_search(population[i], D, max_evals=100_000)
                    population[i] = improved_tour
                    
                # Re-evaluate population fitness after 3-opt modifications
                population_fitnesses = self._compute_fitness_population(population, D)
                
                # Re-check best after 3-opt
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
            #diversity_counts.append(self._count_unique_edges_in_population(population, D))

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        
        # Plot and save diversity evolution
        #self._diversity_plot_unique_edges(diversity_counts)

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
                k = min(1, cand.size)
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
        return fast_mutate_swap(tour, i, j)

    def _mutate_invert(self, tour: np.ndarray) -> np.ndarray:
        """
            Pick two indices at random and invert the path from i to j
            Return the resulting tour 
        """
        i, j = self.rng.choice(tour.size, size=2, replace=False)
        if i > j:
            i, j = j, i
        return fast_mutate_invert(tour, i, j)

    def _mutate_feasible(self, parent: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
            Try up to `feasible_retry` times to produce a mutated child with finite length
            If unsuccessful, return the parent
        """
        ops = (self._mutate_swap, self._mutate_invert)
        for _ in range(self.feasible_retry):
            op = self.rng.choice(ops)
            child = op(parent)
            #child = self._mutate_swap(parent)
            if np.isfinite(self._tour_length(child, D)):
                return child
        return parent
    
    # ==========================
    #  Crossover operators   
    # ==========================
    def _recombine_feasible(self, p1: np.ndarray, p2: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
            Crossover with random choice between OX and ERX.
            If infeasible, try a few random re-shuffles around problematic edges; 
            otherwise fall back to the fitter parent.
        """
        if self.rng.random() < 0.5:
            child = self._crossover_OX(p1, p2)
        else:
            # ERX
            n = p1.size
            # Pre-generate random values for usage inside JIT
            # We need at most N decisions
            rand_vals = self.rng.random(size=n + 50, dtype=np.float64) # float64 default
            child = fast_crossover_ERX(p1, p2, n, rand_vals)

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
        # Get random slice indices using the class RNG (preserving seed control)
        rand_indices = self.rng.choice(n, size=2, replace=False)
        # sort them manually or via numpy
        if rand_indices[0] < rand_indices[1]:
            a, b = rand_indices[0], rand_indices[1]
        else:
            a, b = rand_indices[1], rand_indices[0]
            
        # Call JIT function
        # Ensure suitable types (p1, p2 usually int64 or int32)
        return fast_crossover_OX(p1, p2, n, a, b)

    """def _crossover_ERX(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        
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
        
        n = p1.size
        child = -np.ones(n, dtype=int)
        used = np.zeros(n, dtype=bool)
        
        # Build successor-only adjacency (duplicate if shared)
        succ_edges: Dict[int, List[int]] = {int(c): [] for c in p1}
        def add_parent_successors(parent: np.ndarray):
            for i, c in enumerate(parent):
                c = int(parent[(i + 1) % n])   # successor only
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

        return child"""

    # ===========================
    # ---- Selection helpers ----
    # ===========================
    def _tournament_select_idx(self, fitness: np.ndarray, k: int) -> int:
        """
            Return index of the tournament winner (lowest fitness)
        """
        n = fitness.size
        cand = self.rng.integers(0, n, size=k)
        winner = fast_select_winner(fitness, cand)
        return int(winner)
        
    @staticmethod
    def _compute_fitness_population(population: list[np.ndarray], D: np.ndarray) -> np.ndarray:
        """ 
            Compute objective function for each tour 
            Returns an array of fitnesses 
        """
        # Call the instance method or static method? _tour_length is static.
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
        return fast_tour_length(tour, D, tour.size)

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

    a = r1072969(mu=500, lamb=500, k_tournament=5, mutation_rate=0.8, crossover_rate=0.5)

    a.optimize("./tour1000.csv")
    
    #end_time = time.perf_counter()

    #rss_mb = process.memory_info().rss / (1024**2)  # MB
    #print(f"Elapsed time: {end_time - start_time:.2f}s")
    #print(f"Memory used (RSS): {rss_mb:.2f} MB")

    pass