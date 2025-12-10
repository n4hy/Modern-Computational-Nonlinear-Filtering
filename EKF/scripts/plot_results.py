import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Plot Filtering Results')
    parser.add_argument('csv_file', type=str, help='Path to CSV file')
    parser.add_argument('--title', type=str, default='Filtering Results', help='Plot Title')
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found.")
        sys.exit(1)

    df = pd.read_csv(args.csv_file)

    # State dimension detection
    cols = df.columns
    state_cols = [c for c in cols if c.startswith('x_true_')]
    n = len(state_cols)

    # Setup Plots
    # 1. Trajectory (Component wise)
    # 2. Error (with 3-sigma bounds)

    fig, axes = plt.subplots(n, 2, figsize=(15, 3*n))
    fig.suptitle(args.title, fontsize=16)

    t = df['t']

    for i in range(n):
        # Left: State vs Time
        ax = axes[i, 0]
        ax.plot(t, df[f'x_true_{i}'], 'k-', label='True', linewidth=2)
        ax.plot(t, df[f'x_filt_{i}'], 'r--', label='Filtered')
        ax.plot(t, df[f'x_smooth_{i}'], 'b-.', label='Smoothed')

        ax.set_ylabel(f'State {i}')
        if i == 0:
            ax.legend()
        if i == n-1:
            ax.set_xlabel('Time (s)')

        # Right: Error vs Time with Bounds
        ax_err = axes[i, 1]

        err_filt = df[f'x_true_{i}'] - df[f'x_filt_{i}']
        err_smooth = df[f'x_true_{i}'] - df[f'x_smooth_{i}']

        sigma_filt = np.sqrt(df[f'P_filt_{i}'])
        sigma_smooth = np.sqrt(df[f'P_smooth_{i}'])

        ax_err.plot(t, err_filt, 'r', label='Filt Err', alpha=0.7)
        ax_err.fill_between(t, 3*sigma_filt, -3*sigma_filt, color='r', alpha=0.1, label='Filt 3$\sigma$')

        ax_err.plot(t, err_smooth, 'b', label='Smooth Err', alpha=0.7)
        ax_err.fill_between(t, 3*sigma_smooth, -3*sigma_smooth, color='b', alpha=0.1, label='Smooth 3$\sigma$')

        ax_err.set_ylabel(f'Error {i}')
        ax_err.grid(True)
        if i == 0:
            ax_err.legend()
        if i == n-1:
            ax_err.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = args.csv_file.replace('.csv', '.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # Try to show window if possible (but gracefully handle failure)
    try:
        if os.environ.get('DISPLAY'):
            plt.show()
    except Exception as e:
        print(f"Could not display window: {e}")

if __name__ == "__main__":
    main()
