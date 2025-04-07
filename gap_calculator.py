import argparse

def calculate_gap(obtained_value, optimal_value):
  """Calculates the percentage GAP relative to the optimal value."""
  if optimal_value is None:
    print("Optimal value not provided, cannot calculate GAP.")
    return None
  if optimal_value == 0:
    if obtained_value == 0:
      return 0.0 # Both are zero, gap is zero
    else:
      print("Warning: Optimal value is 0, GAP is undefined or infinite.")
      return float('inf') # Or handle as appropriate

  gap = ((obtained_value - optimal_value) / optimal_value) * 100
  return gap

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate the percentage GAP between an obtained value and an optimal value.")
  parser.add_argument("obtained", type=float, help="The value obtained by the heuristic/algorithm.")
  parser.add_argument("optimal", type=float, help="The known optimal value for the instance.")

  args = parser.parse_args()

  gap = calculate_gap(args.obtained, args.optimal)

  if gap is not None:
    print(f"Obtained Value: {args.obtained}")
    print(f"Optimal Value:  {args.optimal}")
    print(f"GAP:            {gap:.4f}%")