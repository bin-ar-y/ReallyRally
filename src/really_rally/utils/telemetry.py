
from really_rally.utils.feature_store import normalize_row


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
    # Accepts both in-memory feature dicts and CSV rows (whose headers are
    # Title_Case); without this the CSV path silently plotted nothing.
    metrics_data = [normalize_row(row) for row in metrics_data]
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Note: matplotlib not installed. Skipping telemetry plot generation.")
        return
        
    times = []
    arm_energies = []
    whip_ratios = []
    player_heights = []
    t_arms = []
    t_whips = []
    
    for row in metrics_data:
        if bool(row.get('is_valid_pose', False)):
            times.append(float(row.get('frame', 0)) / fps)
            arm_energies.append(float(row.get('smoothed_arm', 0)))
            whip_ratios.append(float(row.get('smoothed_whip', 0)))
            player_heights.append(float(row.get('player_height', 0)))
            t_arms.append(float(row.get('t_arm', 1.2))) # default to 1.2 if missing
            t_whips.append(float(row.get('t_whip', 1.2)))
            
    if not times:
        print(f"Warning: No valid pose data found to plot for {output_path}")
        return
        
    try:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        axs[0].plot(times, arm_energies, label='Smoothed Arm Energy', color='#1f77b4', linewidth=2)
        axs[0].plot(times, t_arms, label='Dynamic Threshold (T_Arm)', color='#d62728', linestyle='--')
        axs[0].set_ylabel('Arm Energy', fontweight='bold')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Arm Energy over Time', fontweight='bold')
        axs[0].grid(True, alpha=0.3)
        
        axs[1].plot(times, whip_ratios, label='Smoothed Whip Ratio', color='#2ca02c', linewidth=2)
        axs[1].plot(times, t_whips, label='Dynamic Threshold (T_Whip)', color='#d62728', linestyle='--')
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


def generate_evaluation_plot(
    metrics_data: list,
    fps: float,
    gt_segments: list,
    pred_segments: list,
    output_path: str
):
    """Generate a telemetry plot with ground-truth and prediction overlays.
    
    Produces a 3-panel plot (Arm Energy, Whip Ratio, Player Scale) with:
      - Green shaded regions for ground-truth rally intervals
      - Blue shaded regions for predicted rally intervals
    
    This makes threshold failures instantly visible:
      - Green with no blue = false negative (missed rally)
      - Blue with no green = false positive (phantom detection)
    
    Args:
        metrics_data: List of frame metric dicts from the processor.
        fps: Video framerate (for frame-to-time conversion).
        gt_segments:   List of (start_sec, end_sec) ground-truth intervals.
        pred_segments: List of (start_sec, end_sec) predicted intervals.
        output_path:   Where to save the PNG.
    """
    metrics_data = [normalize_row(row) for row in metrics_data]
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Note: matplotlib not installed. Skipping evaluation plot.")
        return
    
    times = []
    arm_energies = []
    whip_ratios = []
    player_heights = []
    t_arms = []
    t_whips = []
    z_arms = []
    z_whips = []
    
    for row in metrics_data:
        if bool(row.get('is_valid_pose', False)):
            times.append(float(row.get('frame', 0)) / fps)
            arm_energies.append(float(row.get('smoothed_arm', 0)))
            whip_ratios.append(float(row.get('smoothed_whip', 0)))
            player_heights.append(float(row.get('player_height', 0)))
            t_arms.append(float(row.get('t_arm', 1.2)))
            t_whips.append(float(row.get('t_whip', 1.2)))
            z_arms.append(float(row.get('z_arm', 0)))
            z_whips.append(float(row.get('z_whip', 0)))
    
    if not times:
        print("Warning: No valid pose data found for evaluation plot.")
        return
    
    def _overlay_regions(ax, gt_segs, pred_segs):
        """Add shaded GT (green) and Pred (blue) regions to an axis."""
        for s, e in gt_segs:
            ax.axvspan(s, e, alpha=0.15, color='green', zorder=0)
        for s, e in pred_segs:
            ax.axvspan(s, e, alpha=0.12, color='blue', zorder=0)
    
    try:
        fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
        
        # Panel 1: Arm Energy (raw smoothed)
        axs[0].plot(times, arm_energies, label='Smoothed Arm Energy', color='#1f77b4', linewidth=1.5)
        axs[0].set_ylabel('Arm Energy', fontweight='bold')
        axs[0].set_title('Arm Energy over Time', fontweight='bold')
        axs[0].grid(True, alpha=0.3)
        _overlay_regions(axs[0], gt_segments, pred_segments)
        
        # Panel 2: Whip Ratio (raw smoothed)
        axs[1].plot(times, whip_ratios, label='Smoothed Whip Ratio', color='#2ca02c', linewidth=1.5)
        axs[1].set_ylabel('Whip Ratio', fontweight='bold')
        axs[1].set_title('Whip Ratio over Time', fontweight='bold')
        axs[1].grid(True, alpha=0.3)
        _overlay_regions(axs[1], gt_segments, pred_segments)
        
        # Panel 3: Z-scored signals with adaptive thresholds
        axs[2].plot(times, z_arms, label='Z-Arm', color='#1f77b4', linewidth=1.5, alpha=0.7)
        axs[2].plot(times, z_whips, label='Z-Whip', color='#2ca02c', linewidth=1.5, alpha=0.7)
        axs[2].plot(times, t_arms, label='T_Arm (adaptive)', color='#d62728', linestyle='--', linewidth=1)
        axs[2].plot(times, t_whips, label='T_Whip (adaptive)', color='#ff7f0e', linestyle='--', linewidth=1)
        axs[2].set_ylabel('Z-Score / Threshold', fontweight='bold')
        axs[2].set_title('Z-Scored Signals vs Adaptive Thresholds', fontweight='bold')
        axs[2].legend(loc='upper right', fontsize=8)
        axs[2].grid(True, alpha=0.3)
        _overlay_regions(axs[2], gt_segments, pred_segments)
        
        # Panel 4: Player Scale
        axs[3].plot(times, player_heights, label='Player Height (Scale)', color='#ff7f0e', linewidth=1.5)
        axs[3].axhline(y=0.28, color='#d62728', linestyle='--', label='Max Scale Hint (0.28)')
        axs[3].set_ylabel('Normalized Height', fontweight='bold')
        axs[3].set_xlabel('Time (s)', fontweight='bold')
        axs[3].set_title('Player Scale over Time', fontweight='bold')
        axs[3].grid(True, alpha=0.3)
        _overlay_regions(axs[3], gt_segments, pred_segments)
        
        # Shared legend for GT/Pred overlays
        gt_patch = mpatches.Patch(color='green', alpha=0.3, label='Ground Truth')
        pred_patch = mpatches.Patch(color='blue', alpha=0.3, label='Predicted')
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            handles.extend([gt_patch, pred_patch])
            ax.legend(handles=handles, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        print(f"[Eval] Exported evaluation plot to {output_path}")
    except Exception as e:
        print(f"Failed to export evaluation plot: {e}")

