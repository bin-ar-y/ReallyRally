import csv
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    times = []
    arm_energies = []
    whip_ratios = []
    player_heights = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get('Valid_Pose', 1)) == 1:
                times.append(float(row['Time(s)']))
                arm_energies.append(float(row['Smoothed_Arm']))
                whip_ratios.append(float(row['Smoothed_Whip']))
                player_heights.append(float(row['Player_Height']))
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    axs[0].plot(times, arm_energies, label='Smoothed Arm Energy', color='blue')
    axs[0].axhline(y=1.2, color='r', linestyle='--', label='Threshold (1.2)')
    axs[0].set_ylabel('Arm Energy')
    axs[0].legend()
    axs[0].set_title('Arm Energy over Time')
    
    axs[1].plot(times, whip_ratios, label='Smoothed Whip Ratio', color='green')
    axs[1].axhline(y=1.2, color='r', linestyle='--', label='Threshold (1.2)')
    axs[1].set_ylabel('Whip Ratio')
    axs[1].legend()
    axs[1].set_title('Whip Ratio over Time')
    
    axs[2].plot(times, player_heights, label='Player Height (Scale)', color='orange')
    axs[2].axhline(y=0.28, color='r', linestyle='--', label='Max Scale Hint (0.28)')
    axs[2].set_ylabel('Normalized Height')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].set_title('Player Scale over Time (Walk to Camera Detection)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_metrics.py <input.csv> <output.png>")
        sys.exit(1)
    plot_metrics(sys.argv[1], sys.argv[2])
