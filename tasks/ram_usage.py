import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse

def plot_memory_usage(year):
    # Step 1: Find all .txt files in folders ending with '_{year}' (current directory + subdirs)
    txt_files = list(Path(".").rglob(f"*_{year}/*.txt"))  # Searches recursively

    if not txt_files:
        raise FileNotFoundError("No .txt files found in folders ending with '_{year}'.")

    # Step 2: Read all numerical values from files
    all_values = []
    for file in txt_files:
        with open(file, 'r') as f:
            try:
                values = [float(line.strip()) for line in f if line.strip()]
                all_values.extend(values)
            except ValueError as e:
                print(f"Skipping invalid data in {file}: {e}")

    if not all_values:
        raise ValueError("No valid numerical data found in the files.")

    # Step 3: Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=10, edgecolor='k', alpha=0.7, color='#1f77b4')
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Add mean line for reference
    plt.axvline(np.mean(all_values), color='red', linestyle='--', label=f'Mean: {np.mean(all_values):.2f}')
    plt.legend()

    plt.show()

    # Print key stats
    print(f"Files processed: {len(txt_files)}")
    print(f"Total values: {len(all_values)}")
    print(f"Min: {np.min(all_values):.2f}, Max: {np.max(all_values):.2f}")
    print(f"Mean: {np.mean(all_values):.2f} ± {np.std(all_values):.2f} (SD)")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot Peak Memory Usage distribution")
    
    parser.add_argument('-y', '--year', type=int, required=True,
                      help="year of analysis")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    plot_memory_usage(args.year)