# Flight Scheduling Optimizer

This project provides Python scripts to solve a flight scheduling problem on parallel runways. It uses heuristic algorithms (Greedy, VND, ILS) to find a schedule that minimizes the makespan (the time the last flight finishes), considering sequence-dependent setup times between flights.

## Files

* `schedule_problem.py`: The main script that runs the Greedy, VND, and ILS heuristics to solve the scheduling problem.
* `gap_calculator.py`: A utility script to calculate the percentage difference (GAP) between the makespan found by the heuristic and a known optimal makespan.
* `n6m2_example.txt`, `n10m2_A.txt`: Example input files defining problem instances.
* `solution.txt`: The default output file generated by `schedule_problem.py`, containing the best schedule found and its makespan.

## Requirements

* Python 3.6+

## Usage

### Solving a Scheduling Instance

To run the main scheduling script, use the following command in your terminal, replacing `<instance_file.txt>` with the path to your input file:

```bash
python schedule_problem.py <instance_file.txt> [options]