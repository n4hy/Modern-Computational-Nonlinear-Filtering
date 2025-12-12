import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_results(csv_path, title, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Assume columns: t, x_true, x_est, P_diag
    # Or for multidimensional: t, x0_true, x1_true, ..., x0_est, x1_est, ...

    # Identify state columns
    cols = df.columns
    true_cols = [c for c in cols if "true" in c]
    est_cols = [c for c in cols if "est" in c]

    num_states = len(true_cols)

    plt.figure(figsize=(12, 4 * num_states))

    for i in range(num_states):
        plt.subplot(num_states, 1, i+1)
        plt.plot(df['t'], df[true_cols[i]], 'k-', label='True')
        plt.plot(df['t'], df[est_cols[i]], 'b--', label='Estimate')
        plt.title(f"{title} - State {i}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ekf_csv", help="Path to EKF results CSV")
    parser.add_argument("ukf_csv", help="Path to UKF results CSV")
    parser.add_argument("output_dir", help="Directory to save plots")
    args = parser.parse_args()

    plot_results(args.ekf_csv, "EKF Optimization (NEON)", f"{args.output_dir}/ekf_optimized.png")
    plot_results(args.ukf_csv, "UKF Optimization (NEON)", f"{args.output_dir}/ukf_optimized.png")
