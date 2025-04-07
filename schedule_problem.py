import random
import time
import argparse
import sys
from typing import List, Tuple, Optional

class ProblemInstance:
    """Stores the data for a scheduling problem instance."""
    def __init__(self, num_flights: int, num_runways: int,
                 processing_times: List[int], setup_times: List[List[int]],
                 ready_times: List[int], penalties: List[int]): # Added ready_times and penalties from PDF
        self.num_flights = num_flights
        self.num_runways = num_runways
        self.processing_times = processing_times
        self.setup_times = setup_times
        # NOTE: The C++ implementation focused on makespan minimization and did not use
        # ready_times or penalties. These are stored here for completeness based on the PDF
        # but are NOT used in the current makespan-focused algorithms derived from the C++ code.
        self.ready_times = ready_times
        self.penalties = penalties

    @staticmethod
    def read_instance(file_path: str) -> 'ProblemInstance':
        """Reads an instance file in the format described in the PDF."""
        try:
            with open(file_path, 'r') as f:
                num_flights = int(f.readline().strip())
                num_runways = int(f.readline().strip())

                f.readline() # Skip blank line
                ready_times = list(map(int, f.readline().strip().split()))
                processing_times = list(map(int, f.readline().strip().split())) # 'c' in PDF example
                penalties = list(map(int, f.readline().strip().split())) # 'p' in PDF example

                f.readline() # Skip blank line
                setup_times = []
                for _ in range(num_flights):
                    setup_times.append(list(map(int, f.readline().strip().split())))

                # Basic validation
                if len(ready_times) != num_flights or len(processing_times) != num_flights or len(penalties) != num_flights:
                    raise ValueError("Mismatch in flight data array lengths.")
                if len(setup_times) != num_flights or any(len(row) != num_flights for row in setup_times):
                     raise ValueError("Mismatch in setup times matrix dimensions.")

                return ProblemInstance(num_flights, num_runways, processing_times, setup_times, ready_times, penalties)
        except FileNotFoundError:
            print(f"Error: Instance file not found at {file_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading instance file {file_path}: {e}", file=sys.stderr)
            sys.exit(1)


class Solution:
    """Represents a schedule solution."""
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        # Schedule: List of lists, where each inner list is the sequence of flight indices for a runway
        self.schedule: List[List[int]] = [[] for _ in range(instance.num_runways)]
        # Runway Costs: Completion time for each runway
        self.runway_costs: List[int] = [0] * instance.num_runways
        # Makespan: The maximum completion time across all runways
        self.makespan: int = 0
        # Note: The C++ code focused on makespan. The PDF objective is total penalty.
        # If implementing the PDF objective, add a 'total_penalty' field.

    def calculate_makespan(self) -> int:
        """Calculates the makespan (max runway completion time) for the current schedule."""
        self.runway_costs = [self.calculate_runway_cost(r) for r in range(self.instance.num_runways)]
        self.makespan = max(self.runway_costs) if self.runway_costs else 0
        return self.makespan

    def calculate_runway_cost(self, runway_index: int) -> int:
        """Calculates the completion time of a specific runway."""
        cost = 0
        runway_schedule = self.schedule[runway_index]
        if not runway_schedule:
            return 0

        last_flight_idx = -1
        current_time = 0
        for flight_idx in runway_schedule:
            setup = 0
            if last_flight_idx != -1:
                setup = self.instance.setup_times[last_flight_idx][flight_idx]

            # If using ready times (as per PDF but not C++ makespan logic):
            # start_time = max(current_time + setup, self.instance.ready_times[flight_idx])
            # current_time = start_time + self.instance.processing_times[flight_idx]

            # Using C++ makespan logic (ignores ready times):
            current_time += setup + self.instance.processing_times[flight_idx]

            last_flight_idx = flight_idx
        return current_time

    def copy(self) -> 'Solution':
        """Creates a deep copy of the solution."""
        new_sol = Solution(self.instance)
        new_sol.schedule = [list(runway) for runway in self.schedule] # Deep copy schedule
        new_sol.runway_costs = list(self.runway_costs)
        new_sol.makespan = self.makespan
        return new_sol

    def write_solution(self, file_path: str):
        """Writes the solution to a file in the format specified by the PDF."""
        try:
            with open(file_path, 'w') as f:
                f.write(f"{self.makespan}\n") # Using makespan as the C++ code did
                for runway_schedule in self.schedule:
                    f.write(" ".join(map(str, [flight_idx + 1 for flight_idx in runway_schedule])) + "\n") # Use 1-based indexing for output flights
        except IOError as e:
             print(f"Error writing solution file {file_path}: {e}", file=sys.stderr)

# --- Helper Functions ---

def calculate_runway_cost_static(runway_schedule: List[int], instance: ProblemInstance) -> int:
    """Static version to calculate cost without a Solution object."""
    cost = 0
    if not runway_schedule:
        return 0

    last_flight_idx = -1
    current_time = 0
    for flight_idx in runway_schedule:
        setup = 0
        if last_flight_idx != -1:
            setup = instance.setup_times[last_flight_idx][flight_idx]

        # C++ makespan logic:
        current_time += setup + instance.processing_times[flight_idx]

        last_flight_idx = flight_idx
    return current_time

# --- Greedy Algorithm ---

def greedy_makespan(instance: ProblemInstance) -> Solution:
    """
    Greedy algorithm to minimize makespan.
    Sorts flights by descending processing time (Longest Processing Time - LPT)
    Assigns each flight to the runway where it finishes earliest, considering setup times.
    Matches the logic in the C++ `greedy_algorithm`.
    """
    solution = Solution(instance)

    # Sort flight indices by processing time descending
    sorted_flight_indices = sorted(range(instance.num_flights),
                                   key=lambda i: instance.processing_times[i],
                                   reverse=True)

    # Assign flights
    for flight_idx in sorted_flight_indices:
        best_runway = -1
        min_completion_time = float('inf')

        for r_idx in range(instance.num_runways):
            current_runway_cost = solution.runway_costs[r_idx]
            last_flight_on_runway = -1
            if solution.schedule[r_idx]:
                last_flight_on_runway = solution.schedule[r_idx][-1]

            setup_cost = 0
            if last_flight_on_runway != -1:
                setup_cost = instance.setup_times[last_flight_on_runway][flight_idx]

            completion_time = current_runway_cost + setup_cost + instance.processing_times[flight_idx]

            if completion_time < min_completion_time:
                min_completion_time = completion_time
                best_runway = r_idx

        # Assign to best runway found
        solution.schedule[best_runway].append(flight_idx)
        solution.runway_costs[best_runway] = min_completion_time # Update cost directly

    # Final makespan calculation
    solution.calculate_makespan()
    return solution

# --- Neighborhood Structures (VND Moves) ---

def find_best_swap_intra_runway(solution: Solution) -> Optional[Solution]:
    """
    Neighborhood 1: Swap two flights within the same runway.
    Finds the *best improving* swap across all runways.
    Returns a new Solution object if an improvement is found, else None.
    (Equivalent to C++ swapElementsInline, but checks all runways for simplicity
     instead of only critical path ones).
    """
    instance = solution.instance
    best_improvement = 0 # We want negative improvement (cost reduction)
    best_move = None # (runway_idx, i, j)

    for r_idx in range(instance.num_runways):
        runway = solution.schedule[r_idx]
        n = len(runway)
        if n < 2:
            continue

        original_runway_cost = solution.runway_costs[r_idx]

        for i in range(n):
            for j in range(i + 1, n):
                # Create a temporary swapped runway
                temp_runway = list(runway)
                temp_runway[i], temp_runway[j] = temp_runway[j], temp_runway[i]

                # Calculate cost of the modified runway
                new_runway_cost = calculate_runway_cost_static(temp_runway, instance)
                improvement = new_runway_cost - original_runway_cost

                # Check if this move improves the overall makespan potentially
                # This is an approximation: we only check if it reduces *this* runway's cost,
                # and if this runway is potentially the bottleneck.
                # A more precise check would recalculate the full makespan, but that's slower.
                current_makespan = solution.makespan
                is_critical_or_improving = (original_runway_cost == current_makespan and improvement < 0) or \
                                           (new_runway_cost < current_makespan) # Allows improvement even if not critical

                if improvement < best_improvement and is_critical_or_improving:
                    best_improvement = improvement
                    best_move = (r_idx, i, j)

    if best_move is not None:
        new_solution = solution.copy()
        r_idx, i, j = best_move
        new_solution.schedule[r_idx][i], new_solution.schedule[r_idx][j] = new_solution.schedule[r_idx][j], new_solution.schedule[r_idx][i]
        new_solution.calculate_makespan() # Recalculate full makespan
        # Ensure the move actually improved the overall makespan
        if new_solution.makespan < solution.makespan:
             return new_solution
        else:
             return None # Move didn't help overall makespan
    else:
        return None

def find_best_swap_inter_runway(solution: Solution) -> Optional[Solution]:
    """
    Neighborhood 2: Swap one flight from one runway with one flight from another.
    Finds the swap that results in the best *overall makespan*.
    Returns a new Solution object if an improvement is found, else None.
    (Equivalent to C++ swapColumns)
    """
    instance = solution.instance
    best_makespan = solution.makespan
    best_move = None # (r1_idx, i, r2_idx, j)

    for r1_idx in range(instance.num_runways):
        for r2_idx in range(r1_idx + 1, instance.num_runways):
            runway1 = solution.schedule[r1_idx]
            runway2 = solution.schedule[r2_idx]
            n1 = len(runway1)
            n2 = len(runway2)

            if n1 == 0 or n2 == 0:
                continue

            for i in range(n1):
                for j in range(n2):
                    # Create temporary schedules
                    temp_schedule = [list(r) for r in solution.schedule]
                    temp_schedule[r1_idx][i], temp_schedule[r2_idx][j] = temp_schedule[r2_idx][j], temp_schedule[r1_idx][i]

                    # Calculate costs of affected runways
                    cost1 = calculate_runway_cost_static(temp_schedule[r1_idx], instance)
                    cost2 = calculate_runway_cost_static(temp_schedule[r2_idx], instance)

                    # Calculate potential new makespan
                    current_max = max(cost1, cost2)
                    for r_other in range(instance.num_runways):
                        if r_other != r1_idx and r_other != r2_idx:
                            current_max = max(current_max, solution.runway_costs[r_other])

                    potential_makespan = current_max

                    if potential_makespan < best_makespan:
                        best_makespan = potential_makespan
                        best_move = (r1_idx, i, r2_idx, j)

    if best_move is not None:
        new_solution = solution.copy()
        r1_idx, i, r2_idx, j = best_move
        # Perform the swap
        flight1 = new_solution.schedule[r1_idx][i]
        flight2 = new_solution.schedule[r2_idx][j]
        new_solution.schedule[r1_idx][i] = flight2
        new_solution.schedule[r2_idx][j] = flight1

        new_solution.calculate_makespan() # Recalculate all costs and makespan
        # Double check improvement as calculation was approximate before
        if new_solution.makespan < solution.makespan:
             return new_solution
        else:
            return None
    else:
        return None

def find_best_reinsertion(solution: Solution) -> Optional[Solution]:
    """
    Neighborhood 3: Move a flight from one runway to a position in another runway.
    Finds the move that results in the best *overall makespan*.
    Returns a new Solution object if an improvement is found, else None.
    (Equivalent to C++ reInsertion)
    """
    instance = solution.instance
    best_makespan = solution.makespan
    best_move = None # (r_from, idx_from, r_to, idx_to)

    for r_from in range(instance.num_runways):
        runway_from = solution.schedule[r_from]
        n_from = len(runway_from)
        if n_from == 0:
            continue

        for idx_from in range(n_from):
            flight_to_move = runway_from[idx_from]

            # Create runway_from without the element
            temp_runway_from = runway_from[:idx_from] + runway_from[idx_from+1:]
            cost_from = calculate_runway_cost_static(temp_runway_from, instance)

            for r_to in range(instance.num_runways):
                 runway_to_orig = solution.schedule[r_to]
                 n_to = len(runway_to_orig)

                 # Try inserting at each possible position (including end)
                 for idx_to in range(n_to + 1):
                     # Create temporary runway_to with insertion
                     temp_runway_to = runway_to_orig[:idx_to] + [flight_to_move] + runway_to_orig[idx_to:]
                     cost_to = calculate_runway_cost_static(temp_runway_to, instance)

                     # Calculate potential makespan
                     current_max = 0
                     if r_from == r_to:
                         current_max = calculate_runway_cost_static(temp_runway_to, instance) # Only one runway affected
                     else:
                         current_max = max(cost_from, cost_to)

                     for r_other in range(instance.num_runways):
                          if r_other != r_from and r_other != r_to:
                               current_max = max(current_max, solution.runway_costs[r_other])

                     potential_makespan = current_max

                     if potential_makespan < best_makespan:
                         best_makespan = potential_makespan
                         best_move = (r_from, idx_from, r_to, idx_to)

    if best_move is not None:
        new_solution = solution.copy()
        r_from, idx_from, r_to, idx_to = best_move

        flight_to_move = new_solution.schedule[r_from].pop(idx_from)
        new_solution.schedule[r_to].insert(idx_to, flight_to_move)

        new_solution.calculate_makespan()
        if new_solution.makespan < solution.makespan:
             return new_solution
        else: # Move was not beneficial overall
             return None
    else:
        return None

# --- VND Algorithm ---

def vnd(initial_solution: Solution) -> Solution:
    """
    Variable Neighborhood Descent.
    Applies neighborhood searches iteratively until no further improvement is found.
    Uses the three neighborhood structures defined above.
    """
    current_solution = initial_solution.copy()
    neighborhood_functions = [
        find_best_swap_intra_runway,
        find_best_swap_inter_runway,
        find_best_reinsertion
    ]
    k = 0
    max_k = len(neighborhood_functions)

    while k < max_k:
        # print(f"VND: Trying Neighborhood {k+1}...") # Debug
        search_function = neighborhood_functions[k]
        improved_solution = search_function(current_solution)

        if improved_solution is not None and improved_solution.makespan < current_solution.makespan:
            # print(f"VND: Improvement found in N{k+1}. Makespan: {improved_solution.makespan}") # Debug
            current_solution = improved_solution
            k = 0 # Go back to the first neighborhood
        else:
            # print(f"VND: No improvement in N{k+1}.") # Debug
            k += 1

    return current_solution

# --- ILS Algorithm ---

def perturb(solution: Solution, strength: float = 0.2) -> Solution:
    """
    Perturbation function for ILS.
    Performs a number of random intra-runway swaps.
    The number of swaps depends on the number of flights and the strength parameter.
    (Equivalent to C++ pertubation, but potentially stronger)
    Returns a new perturbed solution.
    """
    instance = solution.instance
    perturbed_solution = solution.copy()
    num_swaps = max(1, int(instance.num_flights * strength / instance.num_runways))

    for _ in range(num_swaps):
        # Choose a random non-empty runway
        non_empty_runways = [r for r, sched in enumerate(perturbed_solution.schedule) if len(sched) >= 2]
        if not non_empty_runways:
            break # Cannot perform swap if no runway has at least 2 flights

        r_idx = random.choice(non_empty_runways)
        runway = perturbed_solution.schedule[r_idx]
        n = len(runway)

        # Choose two distinct random indices
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, n - 1)

        # Swap
        runway[idx1], runway[idx2] = runway[idx2], runway[idx1]

    # Recalculate makespan after all swaps
    perturbed_solution.calculate_makespan()
    return perturbed_solution

def ils(instance: ProblemInstance, max_iterations: int = 10) -> Solution:
    """
    Iterated Local Search.
    Applies perturbation and VND iteratively.
    (Equivalent to C++ ILS)
    """
    print("Starting ILS...")
    # 1. Initial Solution (Greedy + VND)
    print("  Generating initial solution (Greedy)...")
    s0 = greedy_makespan(instance)
    print(f"  Initial Greedy Makespan: {s0.makespan}")
    print("  Applying VND to initial solution...")
    s_best = vnd(s0)
    print(f"  Initial Best Makespan (after VND): {s_best.makespan}")

    # 2. Iterative Improvement
    current_s = s_best
    for i in range(max_iterations):
        print(f"\nILS Iteration {i+1}/{max_iterations}")
        # Perturbation
        print("  Perturbing current best solution...")
        s_perturbed = perturb(current_s)
        print(f"  Makespan after perturbation: {s_perturbed.makespan}")

        # Local Search (VND)
        print("  Applying VND to perturbed solution...")
        s_local_optimum = vnd(s_perturbed)
        print(f"  Makespan after VND: {s_local_optimum.makespan}")

        # Acceptance Criterion (only accept better solutions)
        if s_local_optimum.makespan < s_best.makespan:
            s_best = s_local_optimum
            current_s = s_best # Update current solution for next perturbation
            print(f"  ** New best solution found: Makespan = {s_best.makespan} **")
        else:
            # Keep current_s for the next perturbation if no improvement
            print("  No improvement over best solution.")
            # Optional: Could use current_s = s_local_optimum even if not better (e.g., for diversification)

    print("\nILS Finished.")
    return s_best

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Flight Scheduling Problem using heuristics.")
    parser.add_argument("instance_file", help="Path to the problem instance file.")
    parser.add_argument("-o", "--output_file", default="solution.txt", help="Path to save the final solution file.")
    parser.add_argument("--ils_iterations", type=int, default=10, help="Number of iterations for ILS.")

    args = parser.parse_args()

    # Read Instance
    print(f"Reading instance from: {args.instance_file}")
    instance = ProblemInstance.read_instance(args.instance_file)
    print(f"Instance loaded: {instance.num_flights} flights, {instance.num_runways} runways.")
    print("-" * 30)

    # --- Run Greedy ---
    print("Running Greedy Algorithm...")
    start_time_greedy = time.perf_counter()
    greedy_solution = greedy_makespan(instance)
    end_time_greedy = time.perf_counter()
    greedy_time = (end_time_greedy - start_time_greedy) * 1000 # ms
    print(f"Greedy Makespan: {greedy_solution.makespan}")
    print(f"Greedy Execution Time: {greedy_time:.4f} ms")
    print("-" * 30)

    # --- Run VND ---
    print("Running VND starting from Greedy solution...")
    start_time_vnd = time.perf_counter()
    vnd_solution = vnd(greedy_solution) # Start VND from greedy
    end_time_vnd = time.perf_counter()
    vnd_time = (end_time_vnd - start_time_vnd) * 1000 # ms
    print(f"VND Makespan: {vnd_solution.makespan}")
    print(f"VND Execution Time: {vnd_time:.4f} ms")
    print("-" * 30)

    # --- Run ILS ---
    # Note: ILS internally calls Greedy and VND for its starting point
    print(f"Running ILS ({args.ils_iterations} iterations)...")
    start_time_ils = time.perf_counter()
    ils_solution = ils(instance, max_iterations=args.ils_iterations)
    end_time_ils = time.perf_counter()
    ils_time = (end_time_ils - start_time_ils) * 1000 # ms
    print(f"Final ILS Makespan: {ils_solution.makespan}")
    print(f"Total ILS Execution Time: {ils_time:.4f} ms")
    print("-" * 30)

    # --- Write Final Solution (from ILS) ---
    final_solution = ils_solution
    print(f"Writing final solution (from ILS) to: {args.output_file}")
    final_solution.write_solution(args.output_file)
    print("Done.")

    # Example of how to print the schedule (using 1-based indexing)
    print("\nFinal Schedule (from ILS):")
    for r_idx, runway_sched in enumerate(final_solution.schedule):
        flights_str = " ".join(map(str, [f+1 for f in runway_sched]))
        print(f"  Runway {r_idx+1}: {flights_str} (Cost: {final_solution.runway_costs[r_idx]}) ")