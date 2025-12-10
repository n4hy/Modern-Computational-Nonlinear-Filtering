import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_ukf_results():
    filename = 'ukf_results.csv'
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run the UKF executable first.")
        return

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # CSV Columns based on main.cpp:
    # k,tx,ty,tvx,tvy,mx,my,fx,fy,fvx,fvy,sx,sy,svx,svy

    plt.figure(figsize=(14, 10))

    # 1. 2D Trajectory
    plt.subplot(2, 2, 1)
    plt.plot(df['tx'], df['ty'], 'k-', label='Truth', linewidth=2)
    plt.plot(df['mx'], df['my'], 'r.', label='Measurements', alpha=0.3, markersize=3)
    plt.plot(df['fx'], df['fy'], 'b--', label='UKF Filtered', linewidth=1)
    plt.plot(df['sx'], df['sy'], 'g-', label='Smoothed', linewidth=1.5)
    plt.title('2D Trajectory (Drag Ball)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 2. X Position over time
    plt.subplot(2, 2, 2)
    plt.plot(df['k'], df['tx'], 'k-', label='Truth')
    plt.plot(df['k'], df['fx'], 'b--', label='Filtered')
    plt.plot(df['k'], df['sx'], 'g-', label='Smoothed')
    plt.title('X Position vs Time step')
    plt.xlabel('Step (k)')
    plt.ylabel('X Position (m)')
    plt.legend()
    plt.grid(True)

    # 3. Y Position over time
    plt.subplot(2, 2, 3)
    plt.plot(df['k'], df['ty'], 'k-', label='Truth')
    plt.plot(df['k'], df['fy'], 'b--', label='Filtered')
    plt.plot(df['k'], df['sy'], 'g-', label='Smoothed')
    plt.title('Y Position vs Time step')
    plt.xlabel('Step (k)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)

    # 4. Position Error (Euclidean Distance)
    # Calculate errors
    # Note: Filtered error is valid at all steps.
    # Smoothed error: In the C++ code, we output smoothed values.
    # For the first L steps, smoothed values might be initial guesses or 0 depending on implementation.
    # We'll just plot what's in the CSV.

    err_filt = ((df['fx'] - df['tx'])**2 + (df['fy'] - df['ty'])**2)**0.5
    err_smooth = ((df['sx'] - df['tx'])**2 + (df['sy'] - df['ty'])**2)**0.5

    plt.subplot(2, 2, 4)
    plt.plot(df['k'], err_filt, 'b-', label='Filtered Error', alpha=0.7)
    plt.plot(df['k'], err_smooth, 'g-', label='Smoothed Error', alpha=0.9)
    plt.title('Position Error (Euclidean)')
    plt.xlabel('Step (k)')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_file = 'ukf_results.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_ukf_results()
