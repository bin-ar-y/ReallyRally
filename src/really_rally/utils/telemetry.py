import os

def generate_telemetry_plot(metrics_data: list, fps: float, output_path: str):
    """
    Generates a 3-panel PNG plot of Arm Energy, Whip Ratio, and Player Scale
    over time based on the frame metrics.
    
    Args:
        metrics_data: List of dictionaries containing frame telemetry 
                     (smoothed_arm, smoothed_whip, player_height, etc.)
        fps: The effective framerate of the metrics data
        output_path: Where to save the generated PNG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Note: matplotlib not installed. Skipping telemetry plot generation.")
        return
        
    times = []
    arm_energies = []
    whip_ratios = []
    player_heights = []
    
    for row in metrics_data:
        if bool(row.get('is_valid_pose', False)):
            times.append(float(row.get('frame', 0)) / fps)
            arm_energies.append(float(row.get('smoothed_arm', 0)))
            whip_ratios.append(float(row.get('smoothed_whip', 0)))
            player_heights.append(float(row.get('player_height', 0)))
            
    if not times:
        print(f"Warning: No valid pose data found to plot for {output_path}")
        return
        
    try:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        axs[0].plot(times, arm_energies, label='Smoothed Arm Energy', color='#1f77b4', linewidth=2)
        axs[0].axhline(y=1.2, color='#d62728', linestyle='--', label='Threshold (1.2)')
        axs[0].set_ylabel('Arm Energy', fontweight='bold')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Arm Energy over Time', fontweight='bold')
        axs[0].grid(True, alpha=0.3)
        
        axs[1].plot(times, whip_ratios, label='Smoothed Whip Ratio', color='#2ca02c', linewidth=2)
        axs[1].axhline(y=1.2, color='#d62728', linestyle='--', label='Threshold (1.2)')
        axs[1].set_ylabel('Whip Ratio', fontweight='bold')
        axs[1].legend(loc='upper right')
        axs[1].set_title('Whip Ratio over Time', fontweight='bold')
        axs[1].grid(True, alpha=0.3)
        
        axs[2].plot(times, player_heights, label='Player Height (Scale)', color='#ff7f0e', linewidth=2)
        axs[2].axhline(y=0.28, color='#d62728', linestyle='--', label='Max Scale Hint (0.28)')
        axs[2].set_ylabel('Normalized Height', fontweight='bold')
        axs[2].set_xlabel('Time (s)', fontweight='bold')
        axs[2].legend(loc='upper right')
        axs[2].set_title('Player Scale over Time (Walk to Camera Detection)', fontweight='bold')
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig) # Important to prevent memory leaks in batch processing
        
        print(f"[Debug] Exported telemetry plot to {output_path}")
    except Exception as e:
        print(f"Failed to export telemetry plot: {e}")
