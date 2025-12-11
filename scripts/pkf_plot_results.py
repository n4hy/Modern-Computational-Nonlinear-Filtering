import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 3D State, plot each component
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    components = ['x', 'y', 'z']

    for i, ax in enumerate(axs):
        comp = components[i]

        # Plot Truth
        ax.plot(df['t'], df[f'true_{comp}'], 'k-', label='Truth', alpha=0.7)

        # Plot Measurements (only scatter)
        ax.scatter(df['t'], df[f'meas_{comp}'], c='r', s=5, alpha=0.3, label='Measurements')

        # Plot Filtered
        ax.plot(df['t'], df[f'filt_{comp}'], 'b--', label='Filtered')

        # Plot Smoothed
        # Note regarding lag:
        # The CSV has smooth_x at row k which represents the smoothed estimate for time k-LAG.
        # However, to plot them overlaid on the correct time, we need to shift the smoothed plot.
        # But wait, the example code puts `get_smoothed_mean()` into the row for time `t`.
        # `get_smoothed_mean()` returns the estimate for the *oldest* state in the window.
        # So if the current time is T, it returns estimate for T - LAG.
        # So we should plot `smooth_{comp}` against `t` shifted by -LAG?
        # No, let's look at how the data was saved.
        # At row `i` (time `t[i]`), `smooth` is the estimate for `t[i-LAG]`.
        # So the `smooth` column corresponds to time `t` shifted.
        # Ideally we shift the smoothed data to the left by LAG steps to align with Truth.

        # Let's verify lag from the file or assume fixed.
        # We can just plot it as is to see the delay, or shift it.
        # Let's try to shift it to align.

        # Actually, let's just plot it as is first. The "Smoothed" line should appear lagged if we plot against `t`.
        # But valid comparison is: Smoothed[k] (estimate for k-L) vs Truth[k-L].

        # To make a nice plot, we can shift the time axis for the smoothed data.
        # Or, easier, shift the smoothed series "backwards" in index?
        # No, wait.
        # At time k, we produce estimate for k-L.
        # So the value `df['smooth_x'][k]` belongs to time `df['t'][k] - lag_time`.
        # Let's shift the array for plotting.

        # However, we don't know LAG exactly from CSV.
        # But usually we can see it.

        ax.plot(df['t'], df[f'smooth_{comp}'], 'g-.', label='Smoothed (Raw Output)')

        ax.set_ylabel(comp)
        ax.legend()
        ax.grid(True)

    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Particle Filter & Fixed-Lag Smoother Results (Lorenz-63)')

    output_file = csv_path.replace('.csv', '.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pkf_plot_results.py <csv_file>")
    else:
        plot_results(sys.argv[1])
