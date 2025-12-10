import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    if not os.path.exists('ekf_results.csv'):
        print("ekf_results.csv not found. Run the C++ executable first.")
        return

    df = pd.read_csv('ekf_results.csv')

    # Filter out rows where smooth is 0 (tail end) if desired,
    # but 0 might obscure actual 0 values.
    # Better to just plot everything.

    plt.figure(figsize=(12, 8))

    # Position
    plt.subplot(2, 1, 1)
    plt.plot(df['t'], df['true_pos'], 'k-', label='True Position', linewidth=2)
    plt.plot(df['t'], df['meas'], 'r.', label='Measurements', alpha=0.3)
    plt.plot(df['t'], df['filt_pos'], 'b--', label='EKF Filtered')
    plt.plot(df['t'], df['smooth_pos'], 'g-', label='Fixed-Lag Smoothed', linewidth=2)
    plt.title('Nonlinear Oscillator Position')
    plt.legend()
    plt.grid(True)

    # Velocity
    plt.subplot(2, 1, 2)
    plt.plot(df['t'], df['true_vel'], 'k-', label='True Velocity', linewidth=2)
    plt.plot(df['t'], df['filt_vel'], 'b--', label='EKF Filtered')
    plt.plot(df['t'], df['smooth_vel'], 'g-', label='Fixed-Lag Smoothed', linewidth=2)
    plt.title('Nonlinear Oscillator Velocity')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ekf_results.png')
    print("Plot saved to ekf_results.png")

if __name__ == "__main__":
    plot_results()
