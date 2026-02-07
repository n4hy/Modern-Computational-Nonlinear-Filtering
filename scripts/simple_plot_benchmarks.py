#!/usr/bin/env python3
"""
Simple benchmark visualization without pandas dependency
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import csv
import sys
from pathlib import Path

def read_csv(filename):
    """Read CSV file and return as list of dicts"""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def plot_performance_comparison(data, output_dir):
    """Create performance comparison plots"""

    # Filter data by filter type
    ukf_data = [d for d in data if d['Filter'] == 'UKF']
    srukf_data = [d for d in data if d['Filter'] == 'SRUKF']

    # Extract data
    problems = [d['Problem'] for d in ukf_data]
    ukf_times = [float(d['Avg_Step_Time_ms']) for d in ukf_data]
    srukf_times = [float(d['Avg_Step_Time_ms']) for d in srukf_data]

    ukf_divs = [int(d['Num_Divergences']) for d in ukf_data]
    srukf_divs = [int(d['Num_Divergences']) for d in srukf_data]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution Time
    x = range(len(problems))
    width = 0.35

    ax1.bar([i - width/2 for i in x], ukf_times, width, label='UKF', color='#1f77b4', alpha=0.8)
    ax1.bar([i + width/2 for i in x], srukf_times, width, label='SRUKF', color='#ff7f0e', alpha=0.8)

    ax1.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg Step Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Computational Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('CoupledOscillators10D', 'CoupledOsc\n(10D)').replace('VanDerPol2D', 'VanDerPol\n(2D)').replace('BearingOnly4D', 'BearingOnly\n(4D)') for p in problems], fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i in x:
        speedup = ((ukf_times[i] - srukf_times[i]) / ukf_times[i]) * 100
        ax1.text(i, max(ukf_times[i], srukf_times[i]) * 1.05,
                f'{speedup:.0f}% faster', ha='center', fontsize=9, fontweight='bold', color='green')

    # Plot 2: Divergence Count
    ax2.bar([i - width/2 for i in x], ukf_divs, width, label='UKF', color='#1f77b4', alpha=0.8)
    ax2.bar([i + width/2 for i in x], srukf_divs, width, label='SRUKF', color='#ff7f0e', alpha=0.8)

    ax2.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Divergences', fontsize=12, fontweight='bold')
    ax2.set_title('Numerical Stability Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('CoupledOscillators10D', 'CoupledOsc\n(10D)').replace('VanDerPol2D', 'VanDerPol\n(2D)').replace('BearingOnly4D', 'BearingOnly\n(4D)') for p in problems], fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add improvement annotations
    for i in x:
        if ukf_divs[i] > 0:
            improvement = ((ukf_divs[i] - srukf_divs[i]) / ukf_divs[i]) * 100
            ax2.text(i, max(ukf_divs[i], srukf_divs[i]) * 1.05,
                    f'{improvement:.0f}% better', ha='center', fontsize=9, fontweight='bold', color='green')

    plt.tight_layout()
    output_file = f'{output_dir}/performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_file}')
    plt.close()

def plot_trajectory(filename, output_dir):
    """Plot trajectory comparison"""

    if not Path(filename).exists():
        return

    # Read trajectory data
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 10:
        return

    # Extract data
    times = [float(r['time']) for r in rows]

    # Determine state dimension
    state_cols = [k for k in rows[0].keys() if k.startswith('true_x')]
    n_states = len(state_cols)

    # Create figure
    n_rows = (n_states + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    problem_name = Path(filename).stem.replace('_', ' ').title()
    fig.suptitle(f'{problem_name} - State Trajectories', fontsize=16, fontweight='bold')

    for i in range(n_states):
        ax = axes[i // 2, i % 2]

        true_vals = [float(r[f'true_x{i}']) for r in rows]
        filt_vals = [float(r[f'filt_x{i}']) for r in rows]

        ax.plot(times, true_vals, 'k-', label='True', linewidth=2, alpha=0.8)
        ax.plot(times, filt_vals, 'b--', label='Filtered', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'State x{i}', fontsize=10, fontweight='bold')
        ax.set_title(f'State Dimension {i}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot if odd number of states
    if n_states % 2 == 1:
        axes[-1, -1].axis('off')

    plt.tight_layout()
    output_file = f'{output_dir}/{Path(filename).stem}_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_file}')
    plt.close()

def create_summary_table(data, output_dir):
    """Create a visual summary table"""

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    headers = ['Filter', 'Problem', 'Avg Time\n(ms/step)', 'Divergences', 'Speedup', 'Stability\nImprovement']

    table_data = []
    ukf_data = [d for d in data if d['Filter'] == 'UKF']
    srukf_data = [d for d in data if d['Filter'] == 'SRUKF']

    for ukf_row, srukf_row in zip(ukf_data, srukf_data):
        problem = ukf_row['Problem'].replace('CoupledOscillators10D', 'Coupled Osc (10D)')\
                                     .replace('VanDerPol2D', 'Van der Pol (2D)')\
                                     .replace('BearingOnly4D', 'Bearing-Only (4D)')

        ukf_time = float(ukf_row['Avg_Step_Time_ms'])
        srukf_time = float(srukf_row['Avg_Step_Time_ms'])
        speedup = ((ukf_time - srukf_time) / ukf_time) * 100

        ukf_divs = int(ukf_row['Num_Divergences'])
        srukf_divs = int(srukf_row['Num_Divergences'])
        stability = ((ukf_divs - srukf_divs) / max(ukf_divs, 1)) * 100 if ukf_divs > 0 else 0

        # UKF row
        table_data.append([
            'UKF',
            problem,
            f'{ukf_time:.4f}',
            str(ukf_divs),
            'baseline',
            'baseline'
        ])

        # SRUKF row
        table_data.append([
            'SRUKF',
            '',
            f'{srukf_time:.4f}',
            str(srukf_divs),
            f'+{speedup:.0f}%' if speedup > 0 else f'{speedup:.0f}%',
            f'+{stability:.0f}%' if stability > 0 else f'{stability:.0f}%'
        ])

        # Separator
        table_data.append([''] * 6)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', colWidths=[0.12, 0.25, 0.15, 0.12, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if table_data[i-1][0] == 'UKF':
                table[(i, j)].set_facecolor('#E3F2FD')
            elif table_data[i-1][0] == 'SRUKF':
                table[(i, j)].set_facecolor('#FFF3E0')

            # Highlight improvements
            if j == 4 and table_data[i-1][j].startswith('+'):
                table[(i, j)].set_text_props(weight='bold', color='green')
            if j == 5 and table_data[i-1][j].startswith('+'):
                table[(i, j)].set_text_props(weight='bold', color='green')

    plt.title('UKF vs SRUKF: Comprehensive Comparison', fontsize=16, fontweight='bold', pad=20)

    output_file = f'{output_dir}/summary_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_file}')
    plt.close()

def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = '.'

    summary_file = f'{results_dir}/benchmark_results.csv'

    if not Path(summary_file).exists():
        print(f"Error: {summary_file} not found!")
        return

    print(f"Loading benchmark results from {summary_file}...")
    data = read_csv(summary_file)

    print("\n=== Generating Visualizations ===")

    # Performance comparison
    plot_performance_comparison(data, results_dir)

    # Summary table
    create_summary_table(data, results_dir)

    # Individual trajectories
    print("\nGenerating trajectory plots...")
    trajectory_files = [
        'coupled_osc_ukf.csv',
        'coupled_osc_srukf.csv',
        'vanderpol_ukf.csv',
        'vanderpol_srukf.csv',
        'bearing_ukf.csv',
        'bearing_srukf.csv'
    ]

    for traj_file in trajectory_files:
        full_path = f'{results_dir}/{traj_file}'
        if Path(full_path).exists():
            plot_trajectory(full_path, results_dir)

    print("\n=== Visualization Complete! ===")
    print(f"\nGenerated files in {results_dir}:")
    print("  - performance_comparison.png")
    print("  - summary_table.png")
    print("  - *_plot.png (individual trajectories)")

if __name__ == '__main__':
    main()
