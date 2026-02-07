#!/usr/bin/env python3
"""
Comprehensive benchmark visualization for nonlinear filtering methods
Compares UKF, SRUKF, and their smoother variants across multiple test problems
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_summary_comparison(df, output_dir='.'):
    """Create comparison plots for all metrics across all methods and problems"""

    # Extract unique problems
    problems = df['Problem'].unique()

    # Color scheme
    colors = {
        'UKF': '#1f77b4',
        'SRUKF': '#ff7f0e',
        'UKF+Smoother': '#2ca02c',
        'SRUKF+Smoother': '#d62728'
    }

    # 1. RMSE Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RMSE Comparison Across Problems', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('RMSE_Overall', 'Overall RMSE'),
        ('RMSE_Smoothed_Overall', 'Smoothed RMSE'),
        ('Mean_NEES', 'Mean NEES')
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        x_pos = np.arange(len(problems))
        width = 0.2

        for i, filter_name in enumerate(['UKF', 'SRUKF', 'UKF+Smoother', 'SRUKF+Smoother']):
            filter_data = df[df['Filter'] == filter_name]
            values = []
            for prob in problems:
                prob_data = filter_data[filter_data['Problem'] == prob]
                if len(prob_data) > 0:
                    values.append(prob_data[metric].values[0])
                else:
                    values.append(0)

            ax.bar(x_pos + i*width, values, width, label=filter_name, color=colors[filter_name])

        ax.set_xlabel('Problem', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title)
        ax.set_xticks(x_pos + 1.5*width)
        ax.set_xticklabels(problems, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_rmse_comparison.png', dpi=300, bbox_inches='tight')
    print(f'Saved RMSE comparison to {output_dir}/benchmark_rmse_comparison.png')

    # 2. Performance (Timing) Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Computational Performance Comparison', fontsize=16, fontweight='bold')

    # Average step time
    ax = axes[0]
    x_pos = np.arange(len(problems))
    width = 0.2

    for i, filter_name in enumerate(['UKF', 'SRUKF', 'UKF+Smoother', 'SRUKF+Smoother']):
        filter_data = df[df['Filter'] == filter_name]
        values = []
        for prob in problems:
            prob_data = filter_data[filter_data['Problem'] == prob]
            if len(prob_data) > 0:
                values.append(prob_data['Avg_Step_Time_ms'].values[0])
            else:
                values.append(0)

        ax.bar(x_pos + i*width, values, width, label=filter_name, color=colors[filter_name])

    ax.set_xlabel('Problem', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Average Step Time')
    ax.set_xticks(x_pos + 1.5*width)
    ax.set_xticklabels(problems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total time
    ax = axes[1]
    for i, filter_name in enumerate(['UKF', 'SRUKF', 'UKF+Smoother', 'SRUKF+Smoother']):
        filter_data = df[df['Filter'] == filter_name]
        values = []
        for prob in problems:
            prob_data = filter_data[filter_data['Problem'] == prob]
            if len(prob_data) > 0:
                values.append(prob_data['Total_Time_ms'].values[0])
            else:
                values.append(0)

        ax.bar(x_pos + i*width, values, width, label=filter_name, color=colors[filter_name])

    ax.set_xlabel('Problem', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Total Execution Time')
    ax.set_xticks(x_pos + 1.5*width)
    ax.set_xticklabels(problems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_timing_comparison.png', dpi=300, bbox_inches='tight')
    print(f'Saved timing comparison to {output_dir}/benchmark_timing_comparison.png')

    # 3. Convergence and Stability
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Convergence and Stability Metrics', fontsize=16, fontweight='bold')

    # Convergence time
    ax = axes[0]
    for i, filter_name in enumerate(['UKF', 'SRUKF', 'UKF+Smoother', 'SRUKF+Smoother']):
        filter_data = df[df['Filter'] == filter_name]
        values = []
        for prob in problems:
            prob_data = filter_data[filter_data['Problem'] == prob]
            if len(prob_data) > 0:
                values.append(prob_data['Convergence_Time'].values[0])
            else:
                values.append(0)

        ax.bar(x_pos + i*width, values, width, label=filter_name, color=colors[filter_name])

    ax.set_xlabel('Problem', fontweight='bold')
    ax.set_ylabel('Time (s)', fontweight='bold')
    ax.set_title('Convergence Time')
    ax.set_xticks(x_pos + 1.5*width)
    ax.set_xticklabels(problems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Number of divergences
    ax = axes[1]
    for i, filter_name in enumerate(['UKF', 'SRUKF', 'UKF+Smoother', 'SRUKF+Smoother']):
        filter_data = df[df['Filter'] == filter_name]
        values = []
        for prob in problems:
            prob_data = filter_data[filter_data['Problem'] == prob]
            if len(prob_data) > 0:
                values.append(prob_data['Num_Divergences'].values[0])
            else:
                values.append(0)

        ax.bar(x_pos + i*width, values, width, label=filter_name, color=colors[filter_name])

    ax.set_xlabel('Problem', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Number of Divergences')
    ax.set_xticks(x_pos + 1.5*width)
    ax.set_xticklabels(problems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print(f'Saved convergence comparison to {output_dir}/benchmark_convergence_comparison.png')

def plot_trajectory(filename, output_dir='.'):
    """Plot trajectory results from CSV file"""

    if not Path(filename).exists():
        print(f"Warning: {filename} not found, skipping...")
        return

    df = pd.read_csv(filename)

    # Determine state dimension
    state_cols_true = [col for col in df.columns if col.startswith('true_x')]
    state_dim = len(state_cols_true)

    # Check if smoothed estimates exist
    has_smoothed = any(col.startswith('smooth_x') for col in df.columns)

    # Create subplot grid
    n_rows = (state_dim + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    problem_name = Path(filename).stem.replace('_', ' ').title()
    fig.suptitle(f'{problem_name} - State Trajectories', fontsize=16, fontweight='bold')

    for i in range(state_dim):
        ax = axes[i // 2, i % 2]

        ax.plot(df['time'], df[f'true_x{i}'], 'k-', label='True', linewidth=2)
        ax.plot(df['time'], df[f'filt_x{i}'], 'b--', label='Filtered', linewidth=1.5)

        if has_smoothed:
            ax.plot(df['time'], df[f'smooth_x{i}'], 'r:', label='Smoothed', linewidth=1.5)

        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel(f'State x{i}', fontweight='bold')
        ax.set_title(f'State Dimension {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide extra subplots if state_dim is odd
    if state_dim % 2 == 1:
        axes[-1, -1].axis('off')

    plt.tight_layout()

    output_file = f'{output_dir}/{Path(filename).stem}_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved trajectory plot to {output_file}')

def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = '.'

    # Load benchmark summary
    summary_file = f'{results_dir}/benchmark_results.csv'

    if not Path(summary_file).exists():
        print(f"Error: {summary_file} not found!")
        print("Please run the benchmark executable first.")
        return

    print(f"Loading benchmark results from {summary_file}...")
    df = pd.read_csv(summary_file)

    print("\n=== Benchmark Summary ===")
    print(df.to_string(index=False))

    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_summary_comparison(df, results_dir)

    # Plot individual trajectories
    print("\nGenerating trajectory plots...")
    trajectory_files = [
        'coupled_osc_ukf.csv',
        'coupled_osc_srukf.csv',
        'coupled_osc_ukf_smooth.csv',
        'coupled_osc_srukf_smooth.csv',
        'vanderpol_ukf.csv',
        'vanderpol_srukf.csv',
        'vanderpol_srukf_smooth.csv',
        'bearing_ukf.csv',
        'bearing_srukf.csv',
        'bearing_srukf_smooth.csv'
    ]

    for traj_file in trajectory_files:
        full_path = f'{results_dir}/{traj_file}'
        if Path(full_path).exists():
            plot_trajectory(full_path, results_dir)

    print("\n=== Visualization complete! ===")

if __name__ == '__main__':
    main()
