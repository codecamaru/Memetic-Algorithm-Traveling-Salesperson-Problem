"""
Expected CSV shape (example):
# Student number: r1000000
# Iteration, Elapsed time, Mean value, Best value, Cycle
0,0.13648390769958496,34662.84219191426,28068.94244295934,13,3,1,29, ...
1,0.16279172897338867,31214.457216291055,26239.465126382936,13,3,1,29, ...
...

We only read the first four columns: Iteration, Elapsed time, Mean value, Best value.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_progress_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV while ignoring lines that start with '#'.
    The file can have variable column counts per row due to the trailing Cycle.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read only the first four columns; ignore commented lines and varying-length rows.
    df = pd.read_csv(
        csv_path,
        comment="#",
        header=None,             # no header once comments are skipped
        engine="python",         # tolerant to variable-length rows
        usecols=[0, 1, 2, 3],    # Iteration, Elapsed time, Mean value, Best value
    )

    # Assign column names explicitly
    df.columns = ["Iteration", "Elapsed time", "Mean value", "Best value"]

    # Ensure numeric types (pandas will parse 'inf' as np.inf automatically)
    for col in ["Iteration", "Elapsed time", "Mean value", "Best value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows that failed to parse Iteration or values
    df = df.dropna(subset=["Iteration", "Mean value", "Best value"]).copy()

    # Sort by Iteration in case rows are out of order
    df = df.sort_values("Iteration").reset_index(drop=True)

    return df

def plot_progress(df: pd.DataFrame, out_path: str | None = None) -> None:
    """
    Plot Mean value and Best value vs. Iteration, with legend and grid.
    Optionally save to out_path, and display interactively.
    """
    x = df["Iteration"].to_numpy(dtype=int)
    y_mean = df["Mean value"].to_numpy(dtype=float)
    y_best = df["Best value"].to_numpy(dtype=float)

    # Optional style; falls back silently if unavailable
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_mean, color="tab:blue", lw=2, label="Mean value")
    ax.plot(x, y_best, color="tab:orange", lw=2, label="Best value")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("EA Progress: Mean vs Best")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Tight layout for tidy spacing
    plt.tight_layout()

    # Save if requested; else save a default alongside the CSV
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")
    else:
        default_out = "ea_progress.png"
        plt.savefig(default_out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {default_out}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot EA progress from TSP.csv")
    parser.add_argument("--csv", default="TSP.csv", help="Path to the CSV file")
    parser.add_argument("--out", default=None, help="Output image path (e.g., progress.png)")
    args = parser.parse_args()

    df = load_progress_csv(args.csv)
    plot_progress(df, out_path=args.out)

if __name__ == "__main__":
    main()
