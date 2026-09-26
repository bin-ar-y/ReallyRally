import csv
import os
from typing import List, Dict

from really_rally.processors.base import TennisRallyProcessor
from really_rally.core.models import RallySegment, ServeEvent
from really_rally.core.serve_detector import ServeDetector
from really_rally.utils.media_io import get_video_date
from really_rally.core.config import PROCESS_EVERY_N_FRAMES


def generate_serve_name(video_path: str) -> str:
    """
    Generates name in format: {Date(MMDDYY)}_{OriginalName}_serve
    """
    date_str = get_video_date(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return f"{date_str}_{video_stem}_serve"


class ServePracticeProcessor(TennisRallyProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_folder_name = "serves"
        self.output_csv_name = "serves.csv"
        self.clip_prefix = "_serve_"
        self.serve_events: List[ServeEvent] = []

    def get_output_filename(self) -> str:
        """
        Returns output filename based on video input: {Date(MMDDYY)}_{VideoStem}_serve.mp4
        """
        base_name = generate_serve_name(self.video_path)
        return f"{base_name}.mp4"

    def analyze_segment_data(self, frame_data: List[Dict]) -> List[RallySegment]:
        """
        Detect serves with the pose-based ServeDetector instead of the generic
        arm-energy/whip-ratio gating used by the rally mode.
        """
        if not frame_data:
            return []

        effective_fps = self.fps / PROCESS_EVERY_N_FRAMES
        detector = ServeDetector(effective_fps)
        events = detector.detect(frame_data)

        self.all_frame_metrics.extend(frame_data)
        self.serve_events.extend(events)

        segments = []
        for ev in events:
            segments.append(
                RallySegment(
                    start_frame=int(ev.start_time * self.fps),
                    end_frame=int(ev.end_time * self.fps),
                    start_time=ev.start_time,
                    end_time=ev.end_time,
                    score=ev.score,
                )
            )
        return segments

    def export_clips(self, segments: List[RallySegment]) -> List[str]:
        clip_paths = super().export_clips(segments)
        self._export_serve_events_csv()
        return clip_paths

    def _export_serve_events_csv(self):
        """Write rich per-serve metadata (biomechanical event timestamps).

        Named after the source video: in a batch run every video writes here,
        and a fixed name would leave only the last video's serves on disk.
        """
        if not self.serve_events:
            return

        csv_path = os.path.join(self.output_dir, f"{self.video_stem}_serve_events.csv")
        video_stem = self.video_stem

        header = [
            "Source Video", "Handedness", "Toss Arm",
            "Start Time", "End Time", "Duration",
            "Toss Time", "Contact Time", "Score",
            "Toss Score", "Contact Score", "Follow-Through Score", "Outcome",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for ev in self.serve_events:
                writer.writerow([
                    video_stem,
                    ev.handedness,
                    ev.toss_arm,
                    f"{ev.start_time:.2f}",
                    f"{ev.end_time:.2f}",
                    f"{ev.duration:.2f}",
                    f"{ev.toss_time:.2f}",
                    f"{ev.contact_time:.2f}",
                    f"{ev.score:.3f}",
                    f"{ev.toss_score:.3f}",
                    f"{ev.contact_score:.3f}",
                    f"{ev.follow_through_score:.3f}",
                    ev.outcome,
                ])
        print(f"[Serve] Exported per-serve metadata to {csv_path}")

    def artifacts(self):
        files = super().artifacts()
        events = os.path.join(self.output_dir, f"{self.video_stem}_serve_events.csv")
        if os.path.exists(events):
            files.append(events)
        return files
