"""Debug + Label Studio: the Tk application.

Ownership is strict:

- the main thread owns every widget, the matplotlib canvas, and the ``after()``
  poller;
- workers own decoding and data loading and communicate only through the
  controller.

``_poll_ui_queue`` is the only place ``.after()`` appears, and it reschedules
itself — the same invariant the extraction GUI is held to, so a stray Tk call
from a worker thread cannot slip in.

A recording's results folder (``<output>/<mode>/<video_stem>/``) is the unit of
work: opening it restores the video, telemetry, labels and predictions together,
so a run can be examined without opening files individually.
"""

from __future__ import annotations

import os
from really_rally.studio.movement_panel import MovementPanel
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image

from really_rally.studio.annotation import AnnotationRepository, AnnotationTrack
from really_rally.studio.config_editor import ConfigEditor
from really_rally.studio.controller import StudioController
from really_rally.studio.evaluation import EvaluationEngine, format_metrics, verdict
from really_rally.studio.experiments import ExperimentRepository, summarize_delta
from really_rally.studio.frame_inspector import FrameInspector
from really_rally.studio.label_panel import LabelActions, LabelPanel
from really_rally.studio.predictions import (
    PredictionRepository,
    predictions_from_events,
    predictions_from_segments,
)
from really_rally.studio.replay import DetectorReplayEngine
from really_rally.studio.results_folder import resolve as resolve_results
from really_rally.studio.telemetry import TelemetryRepository, TelemetryView
from really_rally.utils.media_io import frame_to_time

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

SPEEDS = ["0.25x", "0.5x", "1x", "2x", "4x"]


class StudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ReallyRally — Debug + Label Studio")
        self.geometry("1500x950")

        self.controller = StudioController()
        self.telemetry_view: Optional[TelemetryView] = None
        self._impact_image = None
        self._loaded_video = None
        self._photo = None  # keep a reference; Tk drops unreferenced images
        self._video_path: Optional[str] = None

        # Ground truth lives here and is never written by the detector.
        self.annotations = AnnotationRepository()
        self.annotation_track: Optional[AnnotationTrack] = None

        # Detector output and its history — separate from both features and
        # labels, so tuning rewrites predictions without touching either.
        self.predictions = PredictionRepository()
        self.experiments = ExperimentRepository()
        self.evaluator = EvaluationEngine()
        self.replay_engine = DetectorReplayEngine(mode="rally")
        self._features_path: Optional[str] = None
        self._baseline_config = None
        self._last_metrics: Optional[dict] = None
        self._pending_results = None

        self._build_widgets()
        self._bind_keys()
        self._poll_ui_queue()

    # -- layout -----------------------------------------------------------

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(2, weight=1)

        self._build_toolbar()
        self._build_video_panel()
        self._build_inspector()
        self._build_telemetry_area()
        self._build_status()

    def _build_toolbar(self) -> None:
        bar = ctk.CTkFrame(self, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ctk.CTkButton(bar, text="Open results folder…", width=150,
                      command=self.select_results_folder).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(bar, text="Open video…", width=100,
                      command=self.select_video).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(bar, text="Load telemetry…", width=130,
                      command=self.select_telemetry).pack(side="left", padx=4, pady=8)

        ctk.CTkButton(bar, text="Reconstruct stroke…", width=135,
                      command=self.reconstruct_stroke).pack(side="left", padx=4)

        self.lbl_file = ctk.CTkLabel(bar, text="no dataset loaded", anchor="w",
                                     text_color="gray70")
        self.lbl_file.pack(side="left", padx=12)

        self.lbl_time = ctk.CTkLabel(bar, text="t = 0.00s", font=ctk.CTkFont(weight="bold"))
        self.lbl_time.pack(side="right", padx=8)
        self.lbl_frame = ctk.CTkLabel(bar, text="frame 0", text_color="gray70")
        self.lbl_frame.pack(side="right", padx=8)

    def _build_video_panel(self) -> None:
        panel = ctk.CTkFrame(self)
        panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(panel, text="Open a video to begin.\n\n"
                                                    "Scrub the telemetry below to seek.",
                                           text_color="gray60")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        controls = ctk.CTkFrame(panel, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", pady=(4, 6))

        self.btn_play = ctk.CTkButton(controls, text="▶ Play", width=80,
                                      command=self.toggle_play)
        self.btn_play.pack(side="left", padx=4)
        ctk.CTkButton(controls, text="⏮ -1s", width=60,
                      command=lambda: self.controller.step_seconds(-1)).pack(side="left", padx=2)
        ctk.CTkButton(controls, text="◀", width=40,
                      command=lambda: self.controller.step_frames(-1)).pack(side="left", padx=2)
        ctk.CTkButton(controls, text="▶", width=40,
                      command=lambda: self.controller.step_frames(1)).pack(side="left", padx=2)
        ctk.CTkButton(controls, text="+1s ⏭", width=60,
                      command=lambda: self.controller.step_seconds(1)).pack(side="left", padx=2)

        self.speed_var = tk.StringVar(value="1x")
        ctk.CTkOptionMenu(controls, values=SPEEDS, variable=self.speed_var,
                          width=80, command=self._on_speed).pack(side="right", padx=4)

        self.scrub = ctk.CTkSlider(panel, from_=0, to=1, command=self._on_scrub)
        self.scrub.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.scrub.set(0)

    def _build_inspector(self) -> None:
        tabs = ctk.CTkTabview(self)
        tabs.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)

        inspect_tab = tabs.add("Inspect")
        self.inspector = FrameInspector(inspect_tab)
        self.inspector.widget().pack(fill="both", expand=True)

        label_tab = tabs.add("Labels")
        self.label_panel = LabelPanel(label_tab, self.annotations, LabelActions(
            current_time=self.current_time,
            jump_to=self.jump_to_time,
            import_json=self.import_annotations_json,
            export_json=self.export_annotations_json,
            on_change=self.refresh_annotations,
        ))
        self.label_panel.widget().pack(fill="both", expand=True)

        config_tab = tabs.add("Config")
        self.config_editor = ConfigEditor(
            config_tab, on_apply=self.apply_config, on_reset=self.on_config_reset)
        self.config_editor.widget().pack(fill="both", expand=True)

        eval_tab = tabs.add("Evaluate")
        self._build_eval_panel(eval_tab)

        self.grid_rowconfigure(1, weight=3)

    def _build_eval_panel(self, parent) -> None:
        mode_row = ctk.CTkFrame(parent, fg_color="transparent")
        mode_row.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(mode_row, text="Replay mode:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.replay_mode = ctk.StringVar(value="rally")
        ctk.CTkOptionMenu(mode_row, values=["rally", "serve", "feed"],
                          variable=self.replay_mode, width=100,
                          command=self.set_replay_mode).pack(side="left", padx=6)

        ctk.CTkLabel(parent, text="Predictions vs ground truth",
                     font=ctk.CTkFont(size=12, weight="bold"), anchor="w").pack(
            fill="x", padx=8, pady=(6, 2))

        self.lbl_metrics = ctk.CTkLabel(parent, text="Label some intervals first.",
                                        anchor="w", justify="left", wraplength=380,
                                        font=ctk.CTkFont(size=11))
        self.lbl_metrics.pack(fill="x", padx=8)

        self.lbl_verdict = ctk.CTkLabel(parent, text="", anchor="w",
                                        font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_verdict.pack(fill="x", padx=8, pady=(2, 6))

        nav = ctk.CTkFrame(parent, fg_color="transparent")
        nav.pack(fill="x", padx=8)
        ctk.CTkButton(nav, text="◀ Prev disagreement", width=150,
                      command=lambda: self.goto_disagreement(previous=True)).pack(side="left", padx=2)
        ctk.CTkButton(nav, text="Next disagreement ▶", width=150,
                      command=lambda: self.goto_disagreement(previous=False)).pack(side="left", padx=2)

        missed = ctk.CTkFrame(parent, fg_color="transparent")
        missed.pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(missed, text="Next missed", width=120,
                      command=lambda: self.goto_disagreement(False, "missed")).pack(side="left", padx=2)
        ctk.CTkButton(missed, text="Next false", width=120,
                      command=lambda: self.goto_disagreement(False, "false")).pack(side="left", padx=2)

        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkButton(actions, text="Score now", width=110,
                      command=self.evaluate_now).pack(side="left", padx=2)
        ctk.CTkButton(actions, text="Compare vs previous", width=170,
                      command=self.compare_experiments).pack(side="left", padx=2)

        self.lbl_history = ctk.CTkLabel(parent, text="", anchor="w", justify="left",
                                        wraplength=380, font=ctk.CTkFont(size=10),
                                        text_color="gray70")
        self.lbl_history.pack(fill="x", padx=8, pady=(8, 2))

    def _build_telemetry_area(self) -> None:
        self.telemetry_frame = ctk.CTkFrame(self)
        self.telemetry_frame.grid(row=2, column=0, columnspan=2, sticky="nsew",
                                  padx=10, pady=(0, 5))
        self.telemetry_frame.grid_rowconfigure(0, weight=1)
        self.telemetry_frame.grid_columnconfigure(0, weight=1)

        self.lbl_telemetry = ctk.CTkLabel(
            self.telemetry_frame,
            text="Telemetry appears here once a *_features.csv is loaded.",
            text_color="gray60")
        self.lbl_telemetry.grid(row=0, column=0, sticky="nsew")

    def _build_status(self) -> None:
        self.lbl_status = ctk.CTkLabel(self, text="Ready.", anchor="w", padx=10,
                                       font=ctk.CTkFont(size=10))
        self.lbl_status.grid(row=3, column=0, columnspan=2, sticky="ew")

    # -- data loading -----------------------------------------------------

    def select_results_folder(self) -> None:
        """Open a whole results folder: video, telemetry, labels and predictions.

        This is the intended entry point once a run has finished. The pipeline
        writes one folder per recording, so pointing at it restores the entire
        session instead of asking the user to open a video and a CSV separately.
        """
        folder = filedialog.askdirectory(
            title="Select a results folder (e.g. out/serve/IMG_1165)",
            initialdir=os.path.expanduser("~"))
        if not folder:
            return
        self.open_results_folder(folder)

    def open_results_folder(self, folder: str) -> bool:
        found = resolve_results(folder)
        if not found.is_usable:
            messagebox.showwarning(
                "No recording found",
                f"{os.path.basename(folder)} has no video in it.\n\n"
                "Pick the folder for a single recording, for example "
                "<output>/serve/IMG_1165/, or the mode folder if it holds only one.")
            return False

        self._video_path = found.video
        self.lbl_file.configure(text=os.path.basename(found.video))
        self._pending_results = found
        self.controller.open_dataset(found.video, found.features)
        return True

    def select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.mov *.MOV *.avi *.mkv")],
            initialdir=os.path.expanduser("~/Movies"))
        if not path:
            return
        self._video_path = path
        self.lbl_file.configure(text=os.path.basename(path))
        # Auto-pair the sibling telemetry the pipeline writes next to it.
        self.controller.open_dataset(path, self._guess_telemetry(path))

    def reconstruct_stroke(self):
        from really_rally.core.reconstruction import reconstruct
        from really_rally.studio.stroke_selection import choose_stroke

        if self._impact_image is None or self._loaded_video != self._video_path:
            messagebox.showinfo("Select an impact frame", "Open a video and scrub to the stroke first.")
            return
        self.controller.set_playing(False)
        self.btn_play.configure(text="▶ Play")
        frame, image = self._impact_image
        path = self._loaded_video
        selection = choose_stroke(self, image, frame)
        if selection:
            stroke, roi, before, after, footwork = selection
            output = filedialog.askdirectory(title="Save the 3D reconstruction in…")
            if output:
                self.controller.run_job(reconstruct, path, output, frame, stroke, before, after, roi, footwork, portable_html=False)

    def select_telemetry(self) -> None:
        path = filedialog.askopenfilename(
            title="Select telemetry CSV",
            filetypes=[("Telemetry", "*_features.csv *_debug_metrics.csv *.csv")])
        if not path:
            return
        video = self._video_path
        if not video:
            messagebox.showinfo("No video", "Open the video first, then its telemetry.")
            return
        self.controller.open_dataset(video, path)

    @staticmethod
    def _guess_telemetry(video_path: str) -> Optional[str]:
        """Find the telemetry that belongs to a video.

        Checks the results layout first (side by side in the recording's
        folder), then the older stem-prefixed names so previously produced
        output still opens.
        """
        folder = os.path.dirname(video_path) or "."
        stem = os.path.splitext(os.path.basename(video_path))[0]
        for candidate in (
            os.path.join(folder, "features.csv"),
            os.path.join(folder, "debug_metrics.csv"),
            os.path.join(folder, f"{stem}_features.csv"),
            os.path.join(folder, f"{stem}_debug_metrics.csv"),
        ):
            if os.path.exists(candidate):
                return candidate
        return None

    # -- navigation -------------------------------------------------------

    def _on_scrub(self, value) -> None:
        if not self.controller.video_meta:
            return
        self.controller.request_seek(round(float(value)))

    def _on_speed(self, label: str) -> None:
        self.controller.set_speed(float(label.rstrip("x")))

    def toggle_play(self) -> None:
        playing = self.btn_play.cget("text").startswith("▶")
        self.controller.set_playing(playing)
        self.btn_play.configure(text="⏸ Pause" if playing else "▶ Play")

    def seek_to(self, frame: int) -> None:
        self.controller.request_seek(frame)

    # -- keyboard ---------------------------------------------------------

    def _bind_keys(self) -> None:
        # Navigation follows familiar video-editing tools; labelling follows a
        # bracket-delimited interval / single-key point workflow.
        #
        #   space         play / pause
        #   ← →           ∓1 frame          shift+← →   ∓10 frames
        #   j l           ∓1 second         shift+j l   ∓5 seconds
        #   ↑ ↓           previous / next label (or detector event)
        #   [ ]           start / close a ground-truth interval
        #   s             add a point label at the playhead
        #   backspace     delete the label at the playhead
        self.bind("<space>", lambda e: self.toggle_play())
        self.bind("<Left>", lambda e: self.controller.step_frames(-1))
        self.bind("<Right>", lambda e: self.controller.step_frames(1))
        self.bind("<Shift-Left>", lambda e: self.controller.step_frames(-10))
        self.bind("<Shift-Right>", lambda e: self.controller.step_frames(10))
        self.bind("<j>", lambda e: self.controller.step_seconds(-1))
        self.bind("<l>", lambda e: self.controller.step_seconds(1))
        self.bind("<J>", lambda e: self.controller.step_seconds(-5))
        self.bind("<L>", lambda e: self.controller.step_seconds(5))
        self.bind("<Up>", lambda e: self.jump_to_adjacent_label(previous=True))
        self.bind("<Down>", lambda e: self.jump_to_adjacent_label(previous=False))
        self.bind("<bracketleft>", lambda e: self.label_panel.mark_start())
        self.bind("<bracketright>", lambda e: self.label_panel.close_interval())
        self.bind("<s>", lambda e: self.label_panel.add_point())
        self.bind("<BackSpace>", lambda e: self.label_panel.delete_at_playhead())

    # -- time coordinate --------------------------------------------------

    def current_time(self) -> float:
        """The one place a 'current time' is derived. Frame is authoritative."""
        fps = self.controller.video_meta.get("fps") or 30.0
        return frame_to_time(self.controller.current_frame, fps)

    def jump_to_time(self, time_sec: float) -> None:
        fps = self.controller.video_meta.get("fps") or 30.0
        self.seek_to(int(round(time_sec * fps)))

    def jump_to_adjacent_label(self, previous: bool) -> None:
        t = self.current_time()
        target = (self.annotations.previous_label(t - 0.01) if previous
                  else self.annotations.next_label(t + 0.01))
        if target is not None:
            self.jump_to_time(target.span()[0])

    # -- annotation persistence -------------------------------------------

    def _session_dir(self) -> Optional[str]:
        """Where this recording's studio files live: beside its video.

        The pipeline writes a recording's video and telemetry into one folder,
        so labels and predictions go there too — the folder stays self-contained
        and can be reopened (or handed on) as a whole.
        """
        if not self._video_path:
            return None
        return os.path.dirname(os.path.abspath(self._video_path))

    def _labels_path(self) -> Optional[str]:
        folder = self._session_dir()
        return os.path.join(folder, "labels.csv") if folder else None

    def _load_annotations(self) -> None:
        """Load labels for this recording, or start a fresh set."""
        path = self._labels_path()
        if path and os.path.exists(path):
            self.annotations = AnnotationRepository.from_labels_csv(path)
        else:
            self.annotations = AnnotationRepository(path=path)
        self.label_panel.repository = self.annotations
        self.label_panel._pending_start = None
        self.label_panel.refresh()
        self._refresh_annotation_track()

    def import_annotations_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Import ground-truth annotations",
            filetypes=[("Annotations", "annotations.json *.json")])
        if not path:
            return
        imported = AnnotationRepository.from_annotations_json(path)
        if not imported.labels:
            messagebox.showinfo("Import", "That file contains no annotations.")
            return
        # Imported labels replace the set for this recording, and are written to
        # the studio's own file so the original stays untouched as the source.
        self.annotations = AnnotationRepository(imported.labels, path=self._labels_path())
        self.annotations.save()
        self.label_panel.repository = self.annotations
        self.label_panel.refresh()
        self._refresh_annotation_track()
        self.lbl_status.configure(
            text=f"Imported {len(imported.labels)} annotation(s).")

    def export_annotations_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export ground-truth annotations",
            defaultextension=".json", initialfile="annotations.json")
        if not path:
            return
        self.annotations.to_annotations_json(path)
        self.lbl_status.configure(
            text=f"Exported {len(self.annotations.intervals())} interval(s) to "
                 f"{os.path.basename(path)}")

    def _refresh_annotation_track(self) -> None:
        if self.annotation_track is not None:
            self.annotation_track.redraw(self.annotations)

    # -- predictions ------------------------------------------------------

    def _load_predictions(self, video_path: str) -> None:
        """Show the detector's current output, replayed from cached features.

        Replaying rather than reading ``rallies.csv`` is deliberate: the clip
        manifest describes exported clips, which only exist for segments that
        survived export. The feature store replays the detector itself.
        """
        self.predictions = PredictionRepository(path=self._predictions_path())
        if not self._features_path:
            return
        try:
            result = self._run_replay({})
        except Exception as exc:
            self.lbl_status.configure(text=f"Could not replay detection: {exc}")
            return
        self.predictions.replace(self._predictions_for(result))
        if self.telemetry_view is not None:
            self.telemetry_view.set_segment_overlays(
                [p.span() for p in self.predictions.predictions], color="#1f77b4")
        self.evaluate_now()

    def _set_up_tuning(self) -> None:
        """Point the config panel at the configuration the file was made with."""
        from really_rally.core.config_snapshot import ConfigSnapshot

        meta = self.controller.video_meta
        self._baseline_config = ConfigSnapshot.from_module(
            fps=meta.get("fps") or 30.0,
            max_time=(meta.get("frame_count") or 0) / (meta.get("fps") or 30.0),
        )
        self.config_editor.load(self._baseline_config)
        self.experiments.path = self._experiments_path()

    # -- main-thread polling ---------------------------------------------

    def _poll_ui_queue(self) -> None:
        """Drain worker messages and render the newest frame. Main thread only."""
        for message in self.controller.drain_messages():
            self._handle_message(message)

        frame = self.controller.take_frame()
        if frame is not None:
            self._render_frame(*frame)

        self.after(30, self._poll_ui_queue)

    def _handle_message(self, msg) -> None:
        kind = msg[0]
        if kind == "log":
            _, text, _progress = msg
            self.lbl_status.configure(text=text)
        elif kind in ("info", "error", "warning"):
            getattr(messagebox, f"show{kind}")(msg[1], msg[2])
        elif kind == "job-done":
            folder = msg[1]
            self.lbl_status.configure(text=f"3D reconstruction saved in {folder}")
            window = ctk.CTkToplevel(self)
            window.title("Movement analysis · ReallyRally")
            window.geometry("1000x720")
            panel = MovementPanel(window)
            panel.pack(fill="both",expand=True)
            panel.load_folder(folder)
        elif kind == "dataset":
            _, path, meta, repository = msg
            self._on_dataset_loaded(path, meta, repository)

    def _on_dataset_loaded(self, path: str, meta: dict, repository) -> None:
        self._loaded_video = path
        self._impact_image = None
        self.controller.video_meta = meta
        self.controller.repository = repository
        # Prefer what the results folder reported over re-deriving from the
        # filename, so a folder that names its files differently still works.
        discovered = getattr(self._pending_results, "features", None)
        self._features_path = discovered or self._guess_telemetry(path)
        self.scrub.configure(to=max(1, meta["frame_count"] - 1) if meta["frame_count"] else 1)
        self._build_telemetry_view(repository)
        self._load_annotations()
        self._load_predictions(path)
        self._set_up_tuning()
        self.controller.request_seek(0)
        frames = len(repository) if repository else 0
        status = (f"{meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps · "
                  f"{meta['frame_count']} frames · {meta['duration']:.1f}s · "
                  f"{frames} telemetry rows")
        # Say what a results folder is missing, rather than leaving the user to
        # discover that replay silently does nothing.
        found = self._pending_results
        if found is not None and not found.can_replay:
            status += "  —  no features.csv in this folder; re-run extraction to enable replay"
        self.lbl_status.configure(text=status)

    def _build_telemetry_view(self, repository) -> None:
        if self.telemetry_view is not None:
            self.telemetry_view.canvas.get_tk_widget().destroy()
            self.telemetry_view = None
        self.annotation_track = None
        if repository is None or repository.is_empty:
            self.lbl_telemetry.grid()
            self.lbl_telemetry.configure(
                text="No telemetry for this video.\n\n"
                     "Run the pipeline with this video to generate "
                     "*_features.csv, then load it here.")
            return

        self.lbl_telemetry.grid_remove()
        self.telemetry_view = TelemetryView(
            self.telemetry_frame, repository, on_seek=self.seek_to,
            with_label_lane=True)
        self.telemetry_view.toolbar.pack(side="bottom", fill="x")
        self.annotation_track = AnnotationTrack(self.telemetry_view)

    def refresh_annotations(self) -> None:
        """Redraw labels after the panel changed them."""
        self._refresh_annotation_track()

    # -- replay and tuning ------------------------------------------------

    def apply_config(self, overrides: dict) -> None:
        """Re-run detection from cached features with the edited parameters.

        Fast enough to be interactive because it never decodes video: the
        feature store already holds every measurement the detector reads.
        """
        if not self._features_path:
            self.config_editor.set_status(
                "No feature store loaded — run the pipeline for this video first.",
                error=True)
            return
        if not overrides:
            self.config_editor.set_status("No changes to apply.")
            return

        self.config_editor.set_status("Replaying…")
        try:
            result = self._run_replay(overrides)
        except ValueError as exc:
            self.config_editor.set_status(str(exc), error=True)
            return
        except Exception as exc:  # surfaced rather than swallowed
            self.config_editor.set_status(f"Replay failed: {exc}", error=True)
            return

        changed = self._apply_replay_result(result)
        summary = ", ".join(f"{k}={v:g}" for k, v in sorted(overrides.items()))
        self.config_editor.set_status(
            f"{len(result.segments)} segment(s) after {summary}; {changed}")
        self.evaluate_now()

    def _run_replay(self, overrides: dict):
        meta = self.controller.video_meta
        return self.replay_engine.replay(
            self._features_path,
            fps=meta.get("fps"),
            total_frames=meta.get("frame_count"),
            overrides=overrides,
        )

    def set_replay_mode(self, mode: str) -> None:
        """Replay a different detector against the same feature store."""
        self.replay_engine = DetectorReplayEngine(mode=mode)
        self._load_predictions(self._video_path or "")

    @staticmethod
    def _predictions_for(result):
        if result.mode == "serve":
            return predictions_from_events(result.events, result.records)
        return predictions_from_segments(result.segments, result.mode, result.records)

    def _apply_replay_result(self, result) -> str:
        """Swap in the new predictions and redraw the overlays."""
        before = len(self.predictions.predictions)
        predictions = self._predictions_for(result)
        self.predictions.replace(predictions)

        if self.telemetry_view is not None:
            self.telemetry_view.set_segment_overlays(
                [p.span() for p in predictions], color="#1f77b4")
            # The inspector reads the repository, so it must show the replayed
            # telemetry rather than the run that produced the file.
            self.controller.repository = TelemetryRepository(
                result.records, result.config.fps, self.controller.video_meta.get("frame_count"))

        try:
            self.predictions.save(self._predictions_path())
        except OSError:
            pass
        return f"{before} → {len(predictions)} prediction(s)"

    def _predictions_path(self) -> Optional[str]:
        folder = self._session_dir()
        return os.path.join(folder, "predictions.csv") if folder else None

    def _experiments_path(self) -> Optional[str]:
        folder = self._session_dir()
        return os.path.join(folder, "experiments.csv") if folder else None

    def on_config_reset(self) -> None:
        self.lbl_status.configure(text="Configuration reset to the file's values.")

    # -- evaluation -------------------------------------------------------

    def evaluate_now(self) -> None:
        """Score the current predictions against the current labels."""
        gt = self.annotations.interval_tuples()
        pred = self.predictions.spans()
        if not gt and not pred:
            self.lbl_metrics.configure(text="Label some intervals first.")
            self.lbl_verdict.configure(text="")
            return

        metrics = self.evaluator.evaluate(gt, pred)
        metrics["num_gt"] = len(gt)
        metrics["num_pred"] = len(pred)
        metrics["iou_threshold"] = self.evaluator.iou_threshold
        self._last_metrics = metrics

        self.lbl_metrics.configure(text=format_metrics(metrics, self.evaluator.iou_threshold))
        if gt:
            self.lbl_verdict.configure(text=verdict(metrics))
            self.experiments.path = self._experiments_path()
            self.experiments.record(
                self._current_config(), metrics,
                video=self._video_path or "",
                label_set=self._labels_path() or "",
                mode=self.replay_engine.mode,
            )
            self._refresh_history()

    def _current_config(self):
        return self._baseline_config

    def _refresh_history(self) -> None:
        count = len(self.experiments.experiments)
        if not count:
            self.lbl_history.configure(text="")
            return
        latest = self.experiments.experiments[-1]
        text = f"{count} experiment(s) recorded. Latest: {summarize_delta(latest, latest)}"
        self.lbl_history.configure(text=text)

    def compare_experiments(self) -> None:
        """Show what changed between the last two evaluated configurations."""
        from really_rally.studio.experiments import compare

        history = self.experiments.experiments
        if len(history) < 2:
            self.lbl_history.configure(
                text="Need at least two evaluated configurations to compare.")
            return
        baseline, candidate = history[-2], history[-1]
        lines = [summarize_delta(baseline, candidate), ""]
        for label, before, after, direction in compare(baseline, candidate):
            marker = {"better": "▲", "worse": "▼", "same": "="}[direction]
            lines.append(f"{marker} {label}: {before} → {after}")
        self.lbl_history.configure(text="\n".join(lines))

    def goto_disagreement(self, previous: bool, kind: Optional[str] = None) -> None:
        """Jump to the next place the detector and the labels disagree."""
        gt = self.annotations.interval_tuples()
        pred = self.predictions.spans()
        if not gt and not pred:
            self.lbl_status.configure(text="Nothing to compare yet.")
            return
        now = self.current_time()
        finder = (self.evaluator.previous_disagreement if previous
                  else self.evaluator.next_disagreement)
        found = finder(gt, pred, now, kind)
        wrapped = False
        if found is None:
            # Wrap round rather than dead-ending. Stepping "next" when the
            # playhead happens to sit past the only disagreement should return
            # to it, not report that nothing is wrong.
            edge = self.controller.video_meta.get("duration") or 0.0
            found = finder(gt, pred, 0.0 if not previous else edge + 1.0, kind)
            wrapped = found is not None
        if found is None:
            self.lbl_status.configure(text="No disagreements to navigate.")
            return
        self.jump_to_time(found.start)
        self.lbl_status.configure(
            text=f"{'Missed' if found.kind == 'missed' else 'False positive'} at "
                 f"{found.start:.2f}s{' (wrapped)' if wrapped else ''} — {found.detail}")

    def _render_frame(self, frame_idx: int, image) -> None:
        self.controller.current_frame = frame_idx
        fps = self.controller.video_meta.get("fps") or 30.0

        source_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self._impact_image = frame_idx, source_image
        self._photo = ctk.CTkImage(
            light_image=source_image,
            size=(image.shape[1], image.shape[0]))
        self.video_label.configure(image=self._photo, text="")

        self.lbl_time.configure(text=f"t = {frame_to_time(frame_idx, fps):.2f}s")
        self.lbl_frame.configure(text=f"frame {frame_idx}")
        self.scrub.set(frame_idx)

        repository = self.controller.repository
        if repository is not None and not repository.is_empty:
            self.inspector.update(repository.record_at(frame_idx))
            if self.telemetry_view is not None:
                self.telemetry_view.set_playhead(frame_idx)


def main():
    app = StudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
