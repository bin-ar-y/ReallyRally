import os
import sys
import argparse

from really_rally.processors.base import TennisRallyProcessor
from really_rally.processors.serve import ServePracticeProcessor, generate_serve_name
from really_rally.processors.feed import FeedPracticeProcessor, generate_feed_name
from really_rally.utils.media_io import generate_rally_name

def main():
    parser = argparse.ArgumentParser(description="Tennis Video Extractor CLI")
    subparsers = parser.add_subparsers(dest="mode", help="Extraction Mode", required=True)

    # Rally Mode
    parser_rally = subparsers.add_parser("rally", help="Extract generic tennis rallies")
    parser_rally.add_argument("video_path", help="Path to input video file")
    parser_rally.add_argument("--output", help="Path to output directory (default: current directory)", default=".")
    parser_rally.add_argument("--split", action="store_true", help="Keep individual rally clips")
    parser_rally.add_argument("--plot", action="store_true", help="Generate movement telemetry plot")

    # Serve Mode
    parser_serve = subparsers.add_parser("serve", help="Extract tennis serves")
    parser_serve.add_argument("video_path", help="Path to input video file")
    parser_serve.add_argument("--output", help="Path to output directory (default: current directory)", default=".")
    parser_serve.add_argument("--split", action="store_true", help="Keep individual serve clips")
    parser_serve.add_argument("--plot", action="store_true", help="Generate movement telemetry plot")

    # Feed Mode
    parser_feed = subparsers.add_parser("feed", help="Extract multiball training (feeds)")
    parser_feed.add_argument("video_path", help="Path to input video file")
    parser_feed.add_argument("--output", help="Path to output directory (default: current directory)", default=".")
    parser_feed.add_argument("--split", action="store_true", help="Keep individual feed clips")
    parser_feed.add_argument("--plot", action="store_true", help="Generate movement telemetry plot")

    # Evaluate Mode
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate predictions against ground-truth")
    parser_eval.add_argument("--gt", required=True, help="Path to ground-truth JSON annotations")
    parser_eval.add_argument("--pred", required=True, help="Path to predicted rallies CSV")
    parser_eval.add_argument("--iou", type=float, default=0.3, help="Minimum t-IoU threshold (default: 0.3)")
    parser_eval.add_argument("--metrics-csv", help="Path to debug_metrics.csv (for overlay plot)")
    parser_eval.add_argument("--fps", type=float, default=60.0, help="Video FPS for plot time axis (default: 60)")
    parser_eval.add_argument("--plot-output", help="Path to save evaluation overlay PNG")

    args = parser.parse_args()

    # Handle evaluate mode separately (no video_path needed)
    if args.mode == "evaluate":
        from really_rally.utils.evaluate import (
            load_ground_truth, load_predictions, compute_metrics, print_report
        )
        
        gt = load_ground_truth(args.gt)
        pred = load_predictions(args.pred)
        print(f"Loaded {len(gt)} ground-truth segments, {len(pred)} predictions.")
        
        metrics = compute_metrics(gt, pred, args.iou)
        print_report(metrics, args.iou)
        
        # Optional: generate evaluation overlay plot
        if args.plot_output and args.metrics_csv:
            import csv as csv_mod
            metrics_data = []
            with open(args.metrics_csv, 'r') as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    metrics_data.append(row)
            
            if metrics_data:
                from really_rally.utils.telemetry import generate_evaluation_plot
                pred_tuples = [(float(s), float(e)) for s, e in pred]
                generate_evaluation_plot(
                    metrics_data, args.fps, gt, pred_tuples, args.plot_output
                )
        return

    if not os.path.exists(args.video_path):
        print(f"Error: Video not found at {args.video_path}")
        sys.exit(1)

    if args.output == ".":
        if args.mode == "rally":
            base_name = generate_rally_name(args.video_path)
        elif args.mode == "serve":
            base_name = generate_serve_name(args.video_path)
        elif args.mode == "feed":
            base_name = generate_feed_name(args.video_path)
        args.output = os.path.join(".", base_name)

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "rally":
        processor = TennisRallyProcessor(args.video_path, args.output, keep_clips=args.split, generate_plot=args.plot)
    elif args.mode == "serve":
        processor = ServePracticeProcessor(args.video_path, args.output, keep_clips=args.split, generate_plot=args.plot)
    elif args.mode == "feed":
        processor = FeedPracticeProcessor(args.video_path, args.output, keep_clips=args.split, generate_plot=args.plot)

    processor.run()

if __name__ == "__main__":
    main()
