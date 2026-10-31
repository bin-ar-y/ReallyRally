import os
import sys
import argparse

from really_rally.processors.base import TennisRallyProcessor
from really_rally.processors.serve import ServePracticeProcessor
from really_rally.processors.feed import FeedPracticeProcessor

def main():
    parser = argparse.ArgumentParser(description="Tennis Video Extractor CLI")
    subparsers = parser.add_subparsers(dest="mode", help="Extraction Mode", required=True)

    # Rally Mode
    parser_rally = subparsers.add_parser("rally", help="Extract generic tennis rallies")
    parser_rally.add_argument("video_path", help="Path to input video file")
    parser_rally.add_argument("--output", help="Where to write results (default: current directory). Results land in <output>/rally/<video>/", default=".")
    parser_rally.add_argument("--split", action="store_true", help="Keep individual rally clips")
    parser_rally.add_argument("--plot", action="store_true", help="Generate movement telemetry plot")

    # Serve Mode
    parser_serve = subparsers.add_parser("serve", help="Extract tennis serves")
    parser_serve.add_argument("video_path", help="Path to input video file")
    parser_serve.add_argument("--output", help="Where to write results (default: current directory). Results land in <output>/serve/<video>/", default=".")
    parser_serve.add_argument("--split", action="store_true", help="Keep individual serve clips")
    parser_serve.add_argument("--plot", action="store_true", help="Generate movement telemetry plot")

    # Feed Mode
    parser_feed = subparsers.add_parser("feed", help="Extract multiball training (feeds)")
    parser_feed.add_argument("video_path", help="Path to input video file")
    parser_feed.add_argument("--output", help="Where to write results (default: current directory). Results land in <output>/feed/<video>/", default=".")
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

    # Debug + Label Studio
    parser_studio = subparsers.add_parser(
        "studio", help="Open the interactive Debug + Label Studio")
    parser_studio.add_argument("--results",
                               help="Results folder from a run, e.g. out/serve/IMG_1165. "
                                    "Opens its video, telemetry, labels and predictions.")
    parser_studio.add_argument("--video", help="Video to open on launch")
    parser_studio.add_argument("--features", help="Feature store (defaults to features.csv beside the video)")

    args = parser.parse_args()

    if args.mode == "studio":
        from really_rally.studio.app import StudioApp, main as studio_main
        if args.results:
            app = StudioApp()
            if not app.open_results_folder(args.results):
                sys.exit(1)
            app.mainloop()
        elif args.video:
            app = StudioApp()
            app._video_path = args.video
            app.controller.open_dataset(
                args.video, args.features or StudioApp._guess_telemetry(args.video))
            app.mainloop()
        else:
            studio_main()
        return

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

    # Results go to <output>/<mode>/<video_stem>/, so the default keeps the run
    # next to the footage and names the folder after what was extracted rather
    # than after a date-stamped bundle that has to be opened to discover it.
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
