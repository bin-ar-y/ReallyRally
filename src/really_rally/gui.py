"""Extraction workspace. Tk owns the UI; workers own all video and file I/O."""

from collections import deque
from dataclasses import dataclass
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD

from really_rally.processors.base import TennisRallyProcessor
from really_rally.processors.serve import ServePracticeProcessor
from really_rally.processors.feed import FeedPracticeProcessor
from really_rally.utils.media_io import get_video_date, probe_video
from really_rally.utils.subprocesses import ProcessingCancelled

ctk.set_appearance_mode("Dark")

BG = "#111917"
PANEL = "#1A2421"
INSET = "#141D1A"
BORDER = "#33443C"
TEXT = "#EEF3ED"
MUTED = "#A3B4A9"
ACCENT = "#C7E89B"
ACCENT_HOVER = "#B0D67F"
ERROR = "#F2AE99"
MODES = ("Rally Extraction", "Serving Practice", "Multiball Training")
MODE_HINTS = {
    MODES[0]: "Keep the exchanges. Skip the time between points.",
    MODES[1]: "Find serves and group them into a practice reel.",
    MODES[2]: "Pull out strokes from coach-fed practice sessions.",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class ProcessingRequest:
    """An immutable snapshot: workers never read Tk variables or live options."""

    videos: tuple
    output_dir: str
    mode: str
    generate_plot: bool


class RallyExtractorGUI(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("ReallyRally")
        self.geometry("1020x800")
        self.minsize(900, 740)
        self.configure(fg_color=BG)

        self.video_files = []
        self.output_dir = os.getcwd()
        self._output_selected = False
        self.is_processing = False
        self.is_loading = False
        self.stop_event = threading.Event()
        self._ui_queue = queue.Queue()
        self._worker = None
        self._pending_result = None
        self._pending_rows = deque()
        self._load_note = ""
        self._close_requested = False
        self._destroyed = False
        self._poll_id = None
        self._started_at = None
        self._progress = 0.0
        self._result_path = None
        self.processing_mode = tk.StringVar(value=MODES[0])
        self.plot_var = tk.BooleanVar(value=True)

        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.drop_files)
        self.protocol("WM_DELETE_WINDOW", self.request_quit)
        self.bind("<Command-q>" if sys.platform == "darwin" else "<Control-q>", self.request_quit)
        if sys.platform == "darwin":
            self.createcommand("tk::mac::Quit", self.request_quit)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.create_widgets()
        self._poll_ui_queue()

    def _label(self, parent, text, size=13, color=TEXT, **kwargs):
        return ctk.CTkLabel(parent, text=text, text_color=color,
                            font=ctk.CTkFont(family="Avenir Next", size=size), **kwargs)

    def _button(self, parent, text, command, primary=False, **kwargs):
        return ctk.CTkButton(
            parent, text=text, command=command, height=38, corner_radius=9,
            font=ctk.CTkFont(family="Avenir Next", size=13, weight="bold"),
            fg_color=ACCENT if primary else PANEL,
            hover_color=ACCENT_HOVER if primary else BORDER,
            text_color=BG if primary else TEXT, text_color_disabled="#728078",
            border_width=0 if primary else 1, border_color=BORDER, **kwargs)

    def create_widgets(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 20))
        self._label(header, "ReallyRally", size=32).pack(side="left")
        self._label(header, "YOUR PRACTICE, IN FOCUS", size=11, color=ACCENT).pack(side="left", padx=22)
        self._label(header, "LOCAL  /  PRIVATE", size=11, color=MUTED).pack(side="right")

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=28)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2, minsize=320)
        content.grid_rowconfigure(0, weight=1)

        videos = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        videos.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        videos.grid_columnconfigure(0, weight=1)
        videos.grid_rowconfigure(2, weight=1)
        self._label(videos, "01  /  YOUR FOOTAGE", size=11, color=ACCENT, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=22, pady=(20, 4))
        file_header = ctk.CTkFrame(videos, fg_color="transparent")
        file_header.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 12))
        self.lbl_files = self._label(file_header, "Video queue", size=21)
        self.lbl_files.pack(side="left")
        self.btn_clear_files = self._button(file_header, "Clear", self.clear_files, width=65)
        self.btn_clear_files.pack(side="right")

        self.scroll_files = ctk.CTkScrollableFrame(videos, fg_color=INSET, corner_radius=10)
        self.scroll_files.grid(row=2, column=0, sticky="nsew", padx=16)
        self.empty_label = self._label(
            self.scroll_files, "Make room for the good points.\n\nDrop your videos here, or add them below.\nMP4 · MOV · AVI · MKV · M4V",
            size=14, color=MUTED, justify="center", wraplength=340)
        self.empty_label.pack(fill="both", expand=True, padx=12, pady=48)
        self.btn_add_files = self._button(videos, "+  Add videos", self.select_files, width=150)
        self.btn_add_files.grid(row=3, column=0, sticky="ew", padx=22, pady=18)

        settings = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        settings.grid(row=0, column=1, sticky="nsew")
        self._label(settings, "02  /  MAKE IT YOURS", size=11, color=ACCENT, anchor="w").pack(
            fill="x", padx=22, pady=(20, 4))
        self._label(settings, "Session settings", size=21, anchor="w").pack(fill="x", padx=22, pady=(0, 8))
        self.mode_buttons = []
        for mode, title in zip(MODES, ("Rallies", "Serves", "Multiball")):
            button = ctk.CTkRadioButton(
                settings, text=title, variable=self.processing_mode, value=mode,
                command=self._mode_changed, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                border_color=MUTED, text_color=TEXT, font=ctk.CTkFont(family="Avenir Next", size=14))
            button.pack(anchor="w", padx=24, pady=5)
            self.mode_buttons.append(button)
        self.lbl_mode_hint = self._label(settings, MODE_HINTS[MODES[0]], color=MUTED,
                                          justify="left", anchor="w", wraplength=265, height=44)
        self.lbl_mode_hint.pack(fill="x", padx=22, pady=(5, 8))
        ctk.CTkFrame(settings, height=1, fg_color=BORDER).pack(fill="x", padx=22, pady=(0, 10))
        self._label(settings, "Save results to", anchor="w").pack(fill="x", padx=22)
        output_row = ctk.CTkFrame(settings, fg_color="transparent")
        output_row.pack(fill="x", padx=22, pady=(5, 10))
        self.entry_output = ctk.CTkEntry(output_row, fg_color=INSET, border_color=BORDER, text_color=MUTED, height=36)
        self.entry_output.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._show_output()
        self.btn_select_output = self._button(output_row, "Choose", self.select_output, width=74)
        self.btn_select_output.pack(side="right")
        self.chk_plot = ctk.CTkCheckBox(
            settings, text="Include analysis chart", variable=self.plot_var,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, checkmark_color=BG,
            border_color=MUTED, text_color=TEXT, font=ctk.CTkFont(family="Avenir Next", size=13))
        self.chk_plot.pack(anchor="w", padx=24, pady=(0, 6))
        self._label(settings, "Your original videos stay untouched.", size=11, color=MUTED,
                    anchor="w").pack(fill="x", padx=22, pady=(0, 10))

        action = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        action.grid(row=2, column=0, sticky="ew", padx=28, pady=(18, 0))
        action.grid_columnconfigure(0, weight=1)
        status_row = ctk.CTkFrame(action, fg_color="transparent")
        status_row.grid(row=0, column=0, sticky="ew", padx=22, pady=(16, 0))
        self.lbl_progress_text = self._label(status_row, "Ready when you are", size=18)
        self.lbl_progress_text.pack(side="left")
        self.lbl_elapsed = self._label(status_row, "", size=12, color=MUTED)
        self.lbl_elapsed.pack(side="right")
        self.lbl_status = self._label(action, "Add a video to get started.", color=MUTED,
                                      anchor="w", justify="left", wraplength=820, height=42)
        self.lbl_status.grid(row=1, column=0, sticky="ew", padx=22, pady=(2, 8))
        self.progress_bar = ctk.CTkProgressBar(action, height=6, fg_color=BORDER, progress_color=ACCENT)
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=22)
        self.progress_bar.set(0)
        # A separate activity strip stays animated during stages with no percentage,
        # without replacing or inventing the pipeline's measured progress.
        self.activity_bar = ctk.CTkProgressBar(action, height=3, fg_color=PANEL, progress_color=PANEL,
                                               mode="indeterminate", indeterminate_speed=0.6)
        self.activity_bar.grid(row=3, column=0, sticky="ew", padx=22, pady=(5, 12))
        self.activity_bar.set(0)
        buttons = ctk.CTkFrame(action, fg_color="transparent")
        buttons.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 18))
        self.btn_process = self._button(buttons, "Start processing", self.start_processing_thread, primary=True, width=190)
        self.btn_process.pack(side="left")
        self.btn_stop = self._button(buttons, "Stop", self.stop_processing, width=90, state="disabled")
        self.btn_stop.pack(side="left", padx=10)
        self.btn_open_results = self._button(buttons, "Open results", self.open_results, width=130, state="disabled")
        self.btn_open_results.pack(side="right")

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=28, pady=12)
        self._label(footer, "Less footage. More tennis.", size=12, color=MUTED).pack(side="left")
        self.btn_quit = self._button(footer, "Quit", self.request_quit, width=80)
        self.btn_quit.pack(side="right")
        self._set_controls()

    def _busy(self):
        return self.is_processing or self.is_loading

    def _set_controls(self):
        state = "disabled" if self._busy() or self._close_requested else "normal"
        for widget in [self.btn_add_files, self.btn_select_output, self.chk_plot, *self.mode_buttons]:
            widget.configure(state=state)
        self.btn_clear_files.configure(state=state if self.video_files else "disabled")
        self.btn_process.configure(state=state if self.video_files else "disabled")

    def _mode_changed(self):
        self.lbl_mode_hint.configure(text=MODE_HINTS[self.processing_mode.get()])

    def _show_output(self):
        self.entry_output.configure(state="normal")
        self.entry_output.delete(0, "end")
        self.entry_output.insert(0, self.output_dir)
        self.entry_output.xview_moveto(1)
        self.entry_output.configure(state="disabled")

    def drop_files(self, event):
        if self._busy() or self._close_requested or not event.data:
            return
        # Tk's own list parser handles spaces, braces and non-ASCII filenames.
        self._add_files(self.tk.splitlist(event.data))

    def select_files(self):
        if self._busy() or self._close_requested:
            return
        files = filedialog.askopenfilenames(
            parent=self, title="Add practice videos", initialdir=os.path.expanduser("~"),
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.MP4 *.MOV *.AVI *.MKV *.M4V")])
        if files:
            self._add_files(files)

    def _add_files(self, files):
        if self._busy() or self._close_requested:
            return
        self.is_loading = True
        self.stop_event.clear()
        self._started_at = time.monotonic()
        self._set_status("Adding videos…", "Checking your files. You can quit safely at any time.")
        self._progress = 0.0
        self.progress_bar.set(0)
        self.activity_bar.configure(progress_color=ACCENT)
        self.activity_bar.start()
        self._set_controls()
        self.btn_add_files.configure(text="Adding…")
        self._worker = threading.Thread(target=self._validate_files, args=(tuple(files), tuple(self.video_files)), daemon=True)
        self._worker.start()

    def _validate_files(self, files, existing):
        """File-system checks can stall on external drives; keep them off Tk."""
        accepted, skipped = [], 0
        seen = set(existing)
        try:
            for filename in files:
                if self.stop_event.is_set():
                    break
                path = os.path.abspath(filename)
                if path in seen:
                    continue
                if os.path.splitext(path)[1].lower() not in VIDEO_EXTENSIONS or not os.path.isfile(path):
                    skipped += 1
                    continue
                seen.add(path)
                accepted.append(path)
            note = f"{skipped} unsupported or unavailable file(s) skipped." if skipped else ""
            self._ui_queue.put(("files", accepted, note))
        except Exception as exc:
            self._ui_queue.put(("files", accepted, f"Could not add some files: {exc}"))

    def _append_file_row(self, path):
        row = ctk.CTkFrame(self.scroll_files, fg_color=PANEL, corner_radius=8)
        row.pack(fill="x", pady=4, padx=3)
        number = len(self.video_files)
        self._label(row, f"{number:02d}", size=12, color=ACCENT, width=36).pack(side="left", padx=6, pady=12)
        # Truncate the display only; the full source path is retained for processing.
        name = os.path.basename(path)
        if len(name) > 44:
            name = name[:28] + "…" + name[-12:]
        self._label(row, name, anchor="w").pack(side="left", fill="x", expand=True, padx=(0, 12))

    def clear_files(self):
        if self._busy() or self._close_requested:
            return
        self.video_files.clear()
        for widget in self.scroll_files.winfo_children():
            if widget is not self.empty_label:
                widget.destroy()
        self.empty_label.pack(fill="both", expand=True, padx=12, pady=48)
        self.lbl_files.configure(text="Video queue")
        self._set_controls()

    def select_output(self):
        if self._busy() or self._close_requested:
            return
        directory = filedialog.askdirectory(parent=self, title="Save results to")
        if directory:
            self.output_dir = directory
            self._output_selected = True
            self._show_output()

    def _set_status(self, title, detail, color=TEXT):
        self.lbl_progress_text.configure(text=title, text_color=color)
        self.lbl_status.configure(text=detail)

    # All worker messages are consumed here, in bounded batches so a busy worker
    # cannot starve clicks, repaints, the progress animation or the quit flow.
    def log(self, message, progress=None):
        self._ui_queue.put(("log", message, progress))

    def _notify(self, kind, title, message):
        self._ui_queue.put((kind, title, message))

    def _poll_ui_queue(self):
        if self._destroyed:
            return
        latest_log = None
        for _ in range(100):
            try:
                msg = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "log":
                latest_log = msg
            else:
                if latest_log:
                    self._handle_ui_message(latest_log)
                    latest_log = None
                self._handle_ui_message(msg)
        if latest_log:
            self._handle_ui_message(latest_log)

        # Build large selections incrementally so adding hundreds of videos
        # doesn't monopolise the event loop.
        for _ in range(min(8, len(self._pending_rows))):
            path = self._pending_rows.popleft()
            if not self.video_files:
                self.empty_label.pack_forget()
            self.video_files.append(path)
            self._append_file_row(path)
        worker_done = self._worker is None or not self._worker.is_alive()
        if self.is_loading and worker_done and not self._pending_rows and self._pending_result:
            self.is_loading = False
            self._pending_result = None
            if self.video_files and not self._output_selected:
                self.output_dir = os.path.dirname(self.video_files[0])
                self._show_output()
            self.lbl_files.configure(text=f"Video queue · {len(self.video_files)}")
            self.btn_add_files.configure(text="+  Add videos")
            self.activity_bar.stop()
            self.activity_bar.configure(progress_color=PANEL)
            self.activity_bar.set(0)
            self._set_status("Ready when you are", self._load_note or f"{len(self.video_files)} video(s) ready. Choose your session, then start.")
            self._set_controls()
        if self.is_processing and worker_done and self._pending_result:
            result = self._pending_result
            self._pending_result = None
            self.is_processing = False
            self.reset_ui()
            _, status, title, detail, path = result
            self._result_path = path
            self.btn_open_results.configure(state="normal" if path else "disabled")
            if status == "success":
                self._progress = 1.0
                self.progress_bar.set(1)
            self._set_status(title, detail, ERROR if status == "error" else ACCENT)
            self.btn_process.configure(text="Process again" if status == "success" else "Start processing")
            if self._started_at is not None:
                seconds = int(time.monotonic() - self._started_at)
                self.lbl_elapsed.configure(text=f"{seconds // 60:02d}:{seconds % 60:02d} elapsed")
        if self._started_at is not None and self._busy():
            seconds = int(time.monotonic() - self._started_at)
            elapsed = f"{seconds // 60:02d}:{seconds % 60:02d} elapsed"
            percent = f"{self._progress:.0%}  ·  " if self.is_processing else ""
            self.lbl_elapsed.configure(text=percent + elapsed)
        if self._close_requested and worker_done:
            self._destroy_window()
            return
        self._poll_id = self.after(50, self._poll_ui_queue)

    def _handle_ui_message(self, msg):
        kind = msg[0]
        if kind == "log":
            if self.stop_event.is_set():
                return  # don't overwrite the acknowledgement with a late worker update
            _, text, progress = msg
            self.lbl_status.configure(text=text)
            if progress is not None:
                self._progress = max(self._progress, min(0.99, max(0.0, progress)))
                self.progress_bar.set(self._progress)
        elif kind == "files":
            if not self._close_requested:
                self._pending_rows.extend(msg[1])
            self._load_note = msg[2]
            self._pending_result = ("files",)
        elif kind == "finished":
            self._pending_result = msg
        elif kind in {"info", "warning", "error"}:
            self._set_status(msg[1], msg[2], ERROR if kind == "error" else TEXT)

    def start_processing_thread(self):
        if self._busy() or self._close_requested:
            return
        if not self.video_files:
            self._set_status("Add a video first", "Choose Add videos or drop your files into the queue.")
            return
        # Snapshot every Tk variable BEFORE the thread starts.
        request = ProcessingRequest(tuple(self.video_files), self.output_dir,
                                    self.processing_mode.get(), self.plot_var.get())
        self.is_processing = True
        self.stop_event.clear()
        self._pending_result = None
        self._result_path = None
        self._progress = 0.0
        self._started_at = time.monotonic()
        self._set_controls()
        self.btn_process.configure(text="Processing…")
        self.btn_stop.configure(state="normal", text="Stop")
        self.btn_open_results.configure(state="disabled")
        self.progress_bar.set(0)
        self.activity_bar.configure(progress_color=ACCENT)
        self.activity_bar.start()
        self._set_status("Processing your session", "Opening the video and preparing the analysis…")
        self.lbl_elapsed.configure(text="00:00 elapsed")
        self._worker = threading.Thread(target=self.run_processing, args=(request,), daemon=True)
        self._worker.start()

    def stop_processing(self):
        if not self._busy() or self.stop_event.is_set():
            return
        self.stop_event.set()
        self.btn_stop.configure(state="disabled", text="Stopping…")
        self._set_status("Stopping safely…", "Finishing cleanup. Completed clips will stay in your results folder.")

    def request_quit(self, event=None):
        """Window close, application quit and the visible button share one path."""
        if self._close_requested:
            return "break"
        if self._busy():
            if not messagebox.askyesno(
                "Quit ReallyRally?", "Work is still in progress. Stop and quit?\n\n"
                "Completed clips will be kept. The current operation will be cancelled safely.",
                parent=self, default=messagebox.NO, icon=messagebox.QUESTION):
                return "break"
            self._close_requested = True
            self.stop_processing()
            self._pending_rows.clear()
            self._set_status("Closing safely…", "Waiting for the current operation to release its files.")
            self.btn_quit.configure(state="disabled", text="Closing…")
            self._set_controls()
        else:
            self._destroy_window()
        return "break"

    def _destroy_window(self):
        self._destroyed = True
        self.activity_bar.stop()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.destroy()

    def open_results(self):
        if not self._result_path:
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", self._result_path])
            elif sys.platform == "win32":
                os.startfile(self._result_path)
            else:
                subprocess.Popen(["xdg-open", self._result_path])
        except OSError as exc:
            self._set_status("Could not open results", f"{self._result_path}\n{exc}", ERROR)

    def run_processing(self, request):
        """Worker entry point. No Tk calls, including StringVar/BooleanVar.get()."""
        processor = None
        result = ("finished", "stopped", "Processing stopped", "Completed clips are kept. You can start again or quit.", None)
        results_dir = None
        try:
            if self.stop_event.is_set():
                return
            all_clip_paths, serve_events, manifest_rows = [], [], []
            mode = request.mode
            processor_class = {
                MODES[1]: ServePracticeProcessor,
                MODES[2]: FeedPracticeProcessor,
            }.get(mode, TennisRallyProcessor)
            folder = {MODES[1]: "serve", MODES[2]: "feed"}.get(mode, "rally")
            results_dir = os.path.join(request.output_dir, folder)
            self.log("Reading video information…", 0)
            metadata = probe_video(request.videos[0])
            target_format = {key: metadata[key] for key in ("width", "height", "fps")}
            total = len(request.videos)
            for i, video_path in enumerate(request.videos):
                if self.stop_event.is_set():
                    break
                name = os.path.basename(video_path)
                self.log(f"[{i+1}/{total}] Opening {name} and preparing the model…")

                def progress(fraction, message, _i=i, _name=name):
                    # Leave the final 6% for batch merging and metadata.
                    self.log(f"[{_i+1}/{total}] {_name}: {message}", 0.94 * (_i + fraction) / total)

                processor = processor_class(video_path, request.output_dir, keep_clips=True,
                                            progress_callback=progress, stop_event=self.stop_event,
                                            target_format=target_format, generate_plot=request.generate_plot)
                try:
                    if self.stop_event.is_set():
                        return
                    clips = processor.process_video()
                    all_clip_paths.extend(clips)
                    manifest_rows.extend(processor.exported_clip_rows)
                    if mode == MODES[1]:
                        serve_events.extend(processor.serve_events)
                finally:
                    processor.cap.release()
                if self.stop_event.is_set():
                    break
                progress(1, "Video finished")

            if self.stop_event.is_set():
                return
            self.log("Saving the batch manifest…", 0.94)
            TennisRallyProcessor.write_clip_manifest(os.path.join(results_dir, "manifest.csv"), manifest_rows)
            if all_clip_paths:
                self.log(f"Merging {len(all_clip_paths)} clips. This may take a moment…", 0.95)
                date_str = get_video_date(request.videos[0])
                label = {MODES[1]: "Serves", MODES[2]: "Feeds"}.get(mode, "Rallies")
                # Reuse the last processor; merging needs no new model or decoder.
                merged_path = processor.merge_clips(all_clip_paths, f"{date_str}_Merged_{label}.mp4", dest_dir=results_dir)
                if self.stop_event.is_set():
                    return
                if not merged_path:
                    raise RuntimeError("The clips were saved, but the summary could not be merged. Open results to find the clips.")
                self.log("Saving video metadata…", 0.98)
                processor.transfer_metadata(request.videos[0], merged_path)
                if self.stop_event.is_set():
                    return
                clip_type = "feed" if mode == MODES[2] else "rally"
                summary = self._serve_summary(serve_events) if mode == MODES[1] else f"{len(all_clip_paths)} {clip_type} clip(s) saved."
                result = ("finished", "success", "Your session is ready", f"{summary}\nOpen results to review your clips, process again, or quit when you’re ready.", results_dir)
            else:
                label = {MODES[1]: "serves", MODES[2]: "feeds"}.get(mode, "rallies")
                result = ("finished", "success", f"Finished · no {label} found", "Try another session type or video. Available analysis files are in your results folder.", results_dir)
        except ProcessingCancelled:
            pass
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result = ("finished", "error", "Processing needs attention", str(exc), results_dir if results_dir and os.path.isdir(results_dir) else None)
        finally:
            if processor is not None:
                processor.cap.release()
            if self.stop_event.is_set():
                result = ("finished", "stopped", "Processing stopped", "Completed clips are kept. You can start again or quit.", results_dir if results_dir and os.path.isdir(results_dir) else None)
            self._ui_queue.put(result)

    @staticmethod
    def _serve_summary(serve_events):
        if not serve_events:
            return "0 serves detected."
        right = sum(s.handedness == "right" for s in serve_events)
        left = sum(s.handedness == "left" for s in serve_events)
        return f"{len(serve_events)} serve(s) detected · {right} right-handed · {left} left-handed."

    def reset_ui(self):
        self.activity_bar.stop()
        self.activity_bar.configure(progress_color=PANEL)
        self.activity_bar.set(0)
        self._set_controls()
        self.btn_stop.configure(state="disabled", text="Stop")


def main():
    app = RallyExtractorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
