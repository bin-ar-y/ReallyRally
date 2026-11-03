"""Native movement workspace. Tk only presents images; a worker loads and renders them."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import json
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from really_rally.core.session_motion import ReplayResult, read_replay_index
from really_rally.studio.motion_data import METRICS, SOURCES, MotionData, discover

PANEL, INSET, TEXT, MUTED, ACCENT = "#1A2421", "#111C19", "#EEF3ED", "#ABBAAF", "#C7E89B"
GOLD = "#F5B36C"
PICKER = dict(
    fg_color="#30463B",
    button_color="#375744",
    button_hover_color="#456A53",
    dropdown_fg_color=INSET,
    dropdown_hover_color="#375744",
    text_color=TEXT,
)
CHECK = dict(fg_color=ACCENT, hover_color="#AFCF8F", checkmark_color=INSET, border_color=MUTED, text_color=TEXT)


def _button(parent, title, action, width=94):
    return ctk.CTkButton(
        parent,
        text=title,
        command=action,
        width=width,
        height=29,
        fg_color="#30463B",
        hover_color="#46614C",
        text_color=TEXT,
    )


class MovementPanel(ctk.CTkFrame):
    def __init__(self, master, *, analyze=None, stop=None):
        super().__init__(master, fg_color=PANEL, corner_radius=12)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stroke-review")
        self.closed = False
        self.ticket = 0
        self.pending = None
        self.work = None
        self.data = None
        self.entries = {}
        self.active = True
        self.playing = False
        self.when = self.play_origin = self.started = 0.0
        self.yaw = self.pitch = 0.0
        self.drag = None
        self.draw_key = None
        self.pictures = []
        self.frame = 0
        self.grid_columnconfigure((0, 1), weight=1, uniform="views")
        self.grid_rowconfigure(2, weight=1)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=13, pady=(11, 4))
        self.analyze_button = self.stop_button = None
        self.mode = tk.StringVar(self, "Serves")
        self.planted = tk.BooleanVar(self, True)
        if analyze:
            self.analyze_button = _button(bar, "Analyze videos…", analyze, 119)
            self.analyze_button.pack(side="left")
            self.mode_picker = ctk.CTkOptionMenu(
                bar, variable=self.mode, values=["Serves", "Rallies", "Multiball"], width=104, **PICKER
            )
            self.mode_picker.pack(side="left", padx=5)
            self.foot_option = ctk.CTkCheckBox(
                bar, text="Keep a foot on court", variable=self.planted, width=130, **CHECK
            )
            self.foot_option.pack(side="left", padx=5)
            self.stop_button = _button(bar, "Stop", stop, 62)
            self.stop_button.pack(side="left", padx=5)
            self.stop_button.configure(state="disabled")
        _button(bar, "Open session…", self.open_session).pack(side="right", padx=3)
        _button(bar, "Open folder…", self.open_folder).pack(side="right", padx=3)

        clips = ctk.CTkFrame(self, fg_color="transparent")
        clips.grid(row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(5, 8))
        self.clip = tk.StringVar(self, "Choose a recording or open a saved session")
        ctk.CTkLabel(clips, textvariable=self.clip, text_color=TEXT, anchor="w").pack(side="left")
        self.frame_label = ctk.CTkLabel(clips, text="", text_color=ACCENT)
        self.frame_label.pack(side="right")

        self.source = self._view(0, "Source clip")
        self.body = self._view(1, "3D movement   ·   drag to orbit")
        self.body.bind("<ButtonPress-1>", lambda e: setattr(self, "drag", (e.x, e.y)))
        self.body.bind("<B1-Motion>", self.orbit)
        self.body.bind("<ButtonRelease-1>", lambda _e: setattr(self, "drag", None))
        self.view_buttons = ctk.CTkFrame(self, fg_color="transparent")
        self.view_buttons.grid(row=3, column=1, sticky="ew", padx=(4, 14), pady=4)
        for name, yaw, pitch in (("Front", 0, 0), ("Side", np.pi / 2, 0), ("Top", 0, np.pi / 2)):
            _button(self.view_buttons, name, lambda y=yaw, p=pitch: self.view(y, p), 48).pack(side="left", padx=2)
        self.trace = tk.BooleanVar(self, True)
        ctk.CTkCheckBox(
            self.view_buttons, text="Arm trace", variable=self.trace, width=80, command=self.invalidate, **CHECK
        ).pack(side="left", padx=5)
        self.trail = tk.BooleanVar(self, True)
        ctk.CTkCheckBox(
            self.view_buttons, text="Wrist path", variable=self.trail, width=82, command=self.invalidate, **CHECK
        ).pack(side="left", padx=2)
        self.evidence = ctk.CTkLabel(
            self,
            text="Load a session or a saved motion folder.",
            text_color=MUTED,
            anchor="w",
            justify="left",
            wraplength=370,
        )
        self.evidence.grid(row=3, column=0, sticky="ew", padx=15)

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=3)
        self.play = _button(controls, "Play", self.toggle, 67)
        self.play.pack(side="left", padx=2)
        for label, fn in (
            ("−1", lambda: self.seek_index(self.frame - 1)),
            ("+1", lambda: self.seek_index(self.frame + 1)),
            ("Reference", lambda: self.seek_index(self.data.impact) if self.data else None),
        ):
            _button(controls, label, fn, 68).pack(side="left", padx=2)
        self.speed = tk.StringVar(self, "1×")
        ctk.CTkOptionMenu(
            controls, values=["¼×", "½×", "1×"], variable=self.speed, width=65, command=self.set_speed, **PICKER
        ).pack(side="left", padx=4)
        self.scrub = ctk.CTkSlider(
            controls,
            from_=0,
            to=1,
            button_color=ACCENT,
            button_hover_color="#AAC889",
            progress_color="#597A60",
            command=lambda v: self.seek_index(round(v)),
        )
        self.scrub.pack(side="left", fill="x", expand=True, padx=8)
        self.scrub.set(0)

        library = ctk.CTkFrame(self, fg_color=INSET, corner_radius=7)
        library.grid(row=5, column=0, sticky="nsew", padx=(12, 3), pady=(6, 4))
        ctk.CTkLabel(library, text="MOVEMENT CLIPS", text_color=ACCENT, anchor="w").pack(fill="x", padx=10)
        self.clip_list = tk.Listbox(
            library,
            height=4,
            bg=INSET,
            fg=TEXT,
            selectbackground="#345746",
            selectforeground="white",
            activestyle="none",
            exportselection=False,
            borderwidth=0,
            highlightthickness=0,
        )
        scroll = ctk.CTkScrollbar(library, command=self.clip_list.yview, width=10, height=75)
        scroll.pack(side="right", fill="y")
        self.clip_list.configure(yscrollcommand=scroll.set)
        self.clip_list.pack(side="left", fill="both", expand=True, padx=(9, 0), pady=(0, 6))
        self.clip_list.bind("<<ListboxSelect>>", self.choose_list_item)
        signals = ctk.CTkFrame(self, fg_color=INSET, corner_radius=7)
        signals.grid(row=5, column=1, sticky="nsew", padx=(3, 12), pady=(6, 4))
        self.side = tk.StringVar(self, "Right")
        self.metric = tk.StringVar(self, "Elbow flexion")
        ctk.CTkLabel(signals, text="SWING DATA", text_color=ACCENT, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=9
        )
        ctk.CTkOptionMenu(
            signals, values=["Right", "Left"], variable=self.side, width=84, command=self.change_signal, **PICKER
        ).grid(row=1, column=0, padx=(8, 3), pady=2)
        ctk.CTkOptionMenu(
            signals, values=list(METRICS), variable=self.metric, width=149, command=self.change_signal, **PICKER
        ).grid(row=1, column=1, padx=3, sticky="w")
        self.value = ctk.CTkLabel(signals, text="—", text_color=TEXT)
        self.value.grid(row=0, column=2, padx=8, sticky="e")
        self.metrics = ctk.CTkLabel(
            signals, text="", text_color=MUTED, height=22, anchor="w", justify="left", wraplength=375
        )
        self.metrics.grid(row=2, column=0, columnspan=3, sticky="ew", padx=9)
        self.chart = tk.Canvas(signals, height=58, bg=INSET, highlightthickness=0, cursor="crosshair")
        self.chart.grid(row=3, column=0, columnspan=3, sticky="ew", padx=9, pady=(1, 6))
        signals.grid_columnconfigure(2, weight=1)
        self.chart.bind("<Button-1>", self.chart_seek)
        self.chart.bind("<B1-Motion>", self.chart_seek)
        self.chart.bind("<Configure>", lambda _e: self.draw_chart())
        self.note = ctk.CTkLabel(self, text="", anchor="w", text_color=MUTED, height=24)
        self.note.grid(row=6, column=0, columnspan=2, sticky="ew", padx=14)
        self.job_note = ctk.CTkLabel(self, text="", anchor="w", text_color=ACCENT, height=22)
        self.job_note.grid(row=7, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4))
        self._placeholder("Choose Analyze videos or open a saved stroke.")
        self.poll_id = self.after(25, self._poll)

    def _view(self, column, title):
        box = ctk.CTkFrame(self, fg_color=INSET)
        box.grid(row=2, column=column, sticky="nsew", padx=(12, 3) if not column else (3, 12))
        box.grid_columnconfigure(0, weight=1)
        box.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(box, text=title, anchor="w", text_color=TEXT).grid(row=0, column=0, sticky="ew", padx=10)
        canvas = tk.Canvas(box, bg=INSET, highlightthickness=0, width=1, height=1)
        canvas.grid(row=1, column=0, sticky="nsew")
        canvas.bind("<Configure>", lambda _e: self.invalidate())
        return canvas

    def set_busy(self, busy):
        if self.analyze_button:
            state = "disabled" if busy else "normal"
            for widget in (self.analyze_button, self.mode_picker, self.foot_option):
                widget.configure(state=state)
            self.stop_button.configure(state="normal" if busy else "disabled")

    def _placeholder(self, message):
        for canvas in (self.source, self.body):
            canvas.delete("all")
            canvas.create_text(12, 16, text=message, anchor="nw", fill=MUTED)
        self.pictures.clear()

    def open_session(self):
        path = filedialog.askopenfilename(
            parent=self, title="Open saved movement session", filetypes=[("Movement session", "*.json")]
        )
        if path:
            self.load_session(path)

    def open_folder(self):
        path = filedialog.askdirectory(parent=self, title="Open a stroke or results folder")
        if path:
            self.load_folder(path)

    def _load(self, kind, fn, *args):
        self.pause()
        self.ticket += 1
        if self.pending:
            self.pending[1].cancel()
        self.data = None
        self.draw_key = None
        self.frame = 0
        self.when = 0.0
        self.frame_label.configure(text="")
        self.metrics.configure(text="")
        self.value.configure(text="—")
        self.evidence.configure(text="")
        self.chart.delete("all")
        self.note.configure(text="Loading…")
        self._placeholder("Loading…")
        self.pending = kind, self.pool.submit(fn, *args)

    def load_session(self, path):
        self._load("list", read_replay_index, path)

    def load_folder(self, path):
        def list_folder():
            entries = []
            for p in discover(path):
                info = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
                entries.append(ReplayResult(info["source"], info["impact_frame"], info["stroke"], bundle=p))
            return entries

        self._load("list", list_folder)

    def set_entries(self, entries):
        self.entries = {
            f"{i + 1:02d}   {Path(e.source).name}   ·   {e.label}   ·   f{e.frame}": e for i, e in enumerate(entries)
        }
        choices = list(self.entries)
        self.clip_list.delete(0, tk.END)
        for name in choices:
            self.clip_list.insert(tk.END, name)
        choice = next((k for k, v in self.entries.items() if v.bundle), choices[0] if choices else "No movement clips")
        self.clip.set(choice)
        if choices:
            self.clip_list.selection_set(choices.index(choice))
            self.select_clip(choice)
        else:
            self.note.configure(text="No saved movement clips in this folder.")
            self._placeholder("No movement clips")

    def choose_list_item(self, _event):
        selected = self.clip_list.curselection()
        if selected:
            self.select_clip(self.clip_list.get(selected[0]))

    def select_clip(self, choice):
        self.clip.set(choice)
        entry = self.entries[choice]
        if entry.bundle is None:
            self._load("error", lambda: entry.error)
        else:
            self._load("clip", MotionData, entry.bundle)

    def _ready(self, data):
        self.data = data
        self.side.set("Left" if data.meta.get("hitting_hand") == "left" else "Right")
        self.scrub.configure(to=max(1, len(data.frames) - 1))
        self.seek_index(data.impact)
        self.draw_chart()
        ref = {
            "manual": "Selected impact",
            "serve_contact": "Estimated serve contact",
            "motion_peak": "Motion peak",
            "clip_midpoint": "Clip midpoint",
        }.get(data.meta.get("anchor_kind"), "Reference")
        self.note.configure(
            text=f"{ref} · frame {data.frames[data.impact]}     |     Dashed amber = reconstructed motion"
        )
        if not data.source:
            self.note.configure(text="Source recording unavailable. The saved model and swing data are available.")

    def pause(self):
        self.playing = False
        self.play.configure(text="Play")
        if self.data:
            self.when = self.data.times[self.frame]
        self.invalidate()

    def suspend(self):
        self.pause()
        self.active = False

    def toggle(self):
        if not self.data:
            return
        if self.playing:
            self.pause()
        else:
            if self.frame == len(self.data.frames) - 1:
                self.seek_index(0)
            self.started, self.play_origin = time.monotonic(), self.when
            self.playing = True
            self.play.configure(text="Pause")

    def set_speed(self, _value):
        self.started, self.play_origin = time.monotonic(), self.when

    def seek_index(self, index):
        if self.data:
            self.pause()
            self.frame = max(0, min(index, len(self.data.frames) - 1))
            self.when = self.data.times[self.frame]
            self.scrub.set(self.frame)
            self.invalidate()

    def invalidate(self):
        self.draw_key = None

    def orbit(self, event):
        if self.drag:
            dx, dy = event.x - self.drag[0], event.y - self.drag[1]
            self.view(self.yaw + dx * 0.012, np.clip(self.pitch + dy * 0.012, -np.pi / 2, np.pi / 2))
        self.drag = event.x, event.y

    def view(self, yaw, pitch):
        self.yaw, self.pitch = yaw, pitch
        self.invalidate()

    def change_signal(self, _value):
        self.draw_chart()
        self.invalidate()

    def chart_seek(self, event):
        if self.data:
            t = np.clip(event.x / max(1, self.chart.winfo_width()), 0, 1) * self.data.times[-1]
            self.seek_index(int(np.argmin(abs(self.data.times - t))))

    def draw_chart(self):
        self.chart.delete("all")
        if not self.data:
            return
        vals, est, _ = self.data.series(self.side.get(), self.metric.get())
        finite = np.isfinite(vals)
        if not finite.any():
            return
        lo, hi = vals[finite].min(), vals[finite].max()
        times = self.data.times
        x = times * self.chart.winfo_width() / max(times[-1], 0.001)
        y = 52 - (vals - lo) * 46 / max(hi - lo, 0.001)
        gap = min(0.1, 2.5 * np.median(np.diff(times))) if len(times) > 1 else 0
        for i in range(1, len(vals)):
            if finite[i - 1 : i + 1].all() and times[i] - times[i - 1] <= gap:
                inferred = est[i - 1] or est[i]
                self.chart.create_line(
                    x[i - 1],
                    y[i - 1],
                    x[i],
                    y[i],
                    fill=GOLD if inferred else ACCENT,
                    width=2,
                    dash=(4, 3) if inferred else (),
                )
        self.chart.create_line(x[self.data.impact], 0, x[self.data.impact], 57, fill=GOLD, dash=(2, 3))
        self._cursor(self.frame)

    def _cursor(self, i):
        self.chart.delete("cursor")
        x = self.data.times[i] * self.chart.winfo_width() / max(0.001, self.data.times[-1])
        self.chart.create_line(x, 0, x, 58, fill=TEXT, tags="cursor")

    def _present(self, result):
        i, source, body = result
        self.frame = i
        self.scrub.set(i)
        self.pictures.clear()
        for canvas, array in ((self.source, source), (self.body, body)):
            canvas.delete("all")
            if array is None:
                canvas.create_text(10, 20, anchor="nw", text="Source frame unavailable", fill=MUTED)
                continue
            img = Image.fromarray(array)
            w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            ratio = min(w / img.width, h / img.height)
            img = img.resize(
                (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))), Image.Resampling.BILINEAR
            )
            photo = ImageTk.PhotoImage(img, master=canvas)
            self.pictures.append(photo)
            canvas.create_image(w / 2, h / 2, image=photo)
        d, side = self.data, self.side.get()
        self.frame_label.configure(text=f"Frame {d.frames[i]}   ·   {d.times[i] - d.times[d.impact]:+.3f} s")

        def reading(name):
            vals, est, unit = d.series(side, name)
            return ("≈ " if est[i] else "") + f"{vals[i]:.1f}{unit}" if np.isfinite(vals[i]) else "—"

        self.value.configure(text=reading(self.metric.get()))
        self.metrics.configure(
            text="     ".join(
                name + "  " + reading(name)
                for name in ("Arm elevation", "Trunk separation", "Wrist speed", "Knee flexion")
            )
        )
        ids = (11, 13, 15) if side == "Left" else (12, 14, 16)
        self.evidence.configure(
            text="   ·   ".join(
                f"{name}: {SOURCES[d.status[i, j]]}" for name, j in zip(("Shoulder", "Elbow", "Wrist"), ids)
            )
        )
        self._cursor(i)

    def _poll(self):
        if self.closed:
            return
        try:
            if self.pending and self.pending[1].done():
                kind, future = self.pending
                self.pending = None
                value = future.result()
                if kind == "clip":
                    self._ready(value)
                elif kind == "list":
                    self.set_entries(value)
                else:
                    self.note.configure(text=value)
                    self._placeholder("Replay unavailable")
            if self.work and self.work[2].done():
                ticket, key, future = self.work
                self.work = None
                if self.active and ticket == self.ticket and self.data and key == self.draw_key:
                    self._present(future.result())
            if self.active and self.data and self.work is None:
                if self.playing:
                    rate = {"¼×": 0.25, "½×": 0.5, "1×": 1.0}[self.speed.get()]
                    self.when = min(self.data.times[-1], self.play_origin + (time.monotonic() - self.started) * rate)
                    if self.when == self.data.times[-1]:
                        self.playing = False
                        self.play.configure(text="Play")
                width, height = max(1, self.body.winfo_width()), max(1, self.body.winfo_height())
                factor = min(1, 640 / max(width, height))
                size = (max(1, round(width * factor)), max(1, round(height * factor)))
                key = (self.when, size, self.yaw, self.pitch, self.side.get(), self.trace.get(), self.trail.get())
                if key != self.draw_key:
                    self.draw_key = key
                    self.work = self.ticket, key, self.pool.submit(self.data.draw, *key)
        except Exception as exc:
            self.pause()
            self.note.configure(text=f"Could not display movement: {exc}")
            self._placeholder("Replay unavailable")
            self.data = None
        self.poll_id = self.after(16 if self.active else 80, self._poll)

    def destroy(self):
        self.closed = True
        self.after_cancel(self.poll_id)
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().destroy()
