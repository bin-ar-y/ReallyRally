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

    args = parser.parse_args()

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
