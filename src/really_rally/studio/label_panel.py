"""The labelling control panel.

Kept out of ``app.py`` so the widget tree stays a shell and the labelling
workflow is readable in one place. The panel owns no state about the recording:
it calls back into the app for "what time is it" and "record this label", which
keeps the current-time coordinate in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import customtkinter as ctk

from really_rally.studio.annotation import (
    INTERVAL_VOCABULARY,
    POINT_VOCABULARY,
    AnnotationRepository,
)


@dataclass
class LabelActions:
    """Everything the panel needs from the app, passed explicitly."""
    current_time: Callable[[], float]
    jump_to: Callable[[float], None]
    import_json: Callable[[], None]
    export_json: Callable[[], None]
    # Called after any edit so the app can redraw the label track.
    on_change: Callable[[], None]


class LabelPanel:
    """Create, review and delete ground-truth labels."""

    def __init__(self, parent, repository: AnnotationRepository, actions: LabelActions):
        self.repository = repository
        self.actions = actions
        self._pending_start: Optional[float] = None
        self._rows: List = []
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._build()

    def widget(self):
        return self.frame

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        self.lbl_pending = ctk.CTkLabel(
            self.frame, text="Interval: not started", anchor="w",
            font=ctk.CTkFont(size=10), text_color="gray70")
        self.lbl_pending.pack(fill="x", padx=8, pady=(6, 2))

        intervals = ctk.CTkFrame(self.frame, fg_color="transparent")
        intervals.pack(fill="x", padx=8, pady=2)
        self.interval_class = ctk.CTkOptionMenu(
            intervals, values=INTERVAL_VOCABULARY, width=150)
        self.interval_class.pack(side="left")
        ctk.CTkButton(intervals, text="Mark start  [", width=100,
                      command=self.mark_start).pack(side="left", padx=4)
        ctk.CTkButton(intervals, text="Close  ]", width=80,
                      command=self.close_interval).pack(side="left", padx=2)

        points = ctk.CTkFrame(self.frame, fg_color="transparent")
        points.pack(fill="x", padx=8, pady=2)
        self.point_class = ctk.CTkOptionMenu(points, values=POINT_VOCABULARY, width=150)
        self.point_class.pack(side="left")
        ctk.CTkButton(points, text="Add point  s", width=110,
                      command=self.add_point).pack(side="left", padx=4)
        ctk.CTkButton(points, text="Delete at playhead  ⌫", width=170,
                      fg_color="#8B3A3A", hover_color="#6E2E2E",
                      command=self.delete_at_playhead).pack(side="left", padx=2)

        io = ctk.CTkFrame(self.frame, fg_color="transparent")
        io.pack(fill="x", padx=8, pady=(4, 2))
        ctk.CTkButton(io, text="Import annotations.json", width=170,
                      command=lambda: self.actions.import_json()).pack(side="left", padx=2)
        ctk.CTkButton(io, text="Export annotations.json", width=170,
                      command=lambda: self.actions.export_json()).pack(side="left", padx=2)

        self.lbl_count = ctk.CTkLabel(self.frame, text="", anchor="w",
                                      font=ctk.CTkFont(size=11, weight="bold"))
        self.lbl_count.pack(fill="x", padx=8, pady=(6, 2))

        self.list_frame = ctk.CTkScrollableFrame(self.frame, label_text="Labels")
        self.list_frame.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        self.refresh()

    # -- actions ----------------------------------------------------------

    def mark_start(self) -> None:
        self._pending_start = self.actions.current_time()
        self._update_pending_label()

    def close_interval(self) -> None:
        if self._pending_start is None:
            return
        end = self.actions.current_time()
        if end < self._pending_start:
            self._pending_start, end = end, self._pending_start
        self.repository.add_interval(self._pending_start, end,
                                     self.interval_class.get())
        self._pending_start = None
        self._update_pending_label()
        self.refresh()
        self.actions.on_change()

    def add_point(self) -> None:
        self.repository.add_point(self.actions.current_time(), self.point_class.get())
        self.refresh()
        self.actions.on_change()

    def delete_at_playhead(self) -> None:
        if self.repository.remove_at(self.actions.current_time()):
            self.refresh()
            self.actions.on_change()

    @property
    def pending_start(self) -> Optional[float]:
        return self._pending_start

    def cancel_pending(self) -> None:
        self._pending_start = None
        self._update_pending_label()

    # -- rendering --------------------------------------------------------

    def _update_pending_label(self) -> None:
        if self._pending_start is None:
            self.lbl_pending.configure(text="Interval: not started", text_color="gray70")
        else:
            label = self.interval_class.get()
            self.lbl_pending.configure(
                text=f"Interval start at {self._pending_start:.2f}s — "
                     f"play on, then press ] to close as “{label}”",
                text_color="#ffb454")

    def refresh(self) -> None:
        for row in self._rows:
            row.destroy()
        self._rows = []

        counts = self.repository.count()
        self.lbl_count.configure(
            text=f"{counts['intervals']} interval(s) · {counts['points']} point(s)")

        entries = sorted(self.repository.labels, key=lambda entry: entry.span()[0])
        if not entries:
            hint = ctk.CTkLabel(
                self.list_frame,
                text="No labels yet.\n\n[ starts an interval, ] closes it,\n"
                     "s selects a point class and adds it.",
                text_color="gray60", justify="left", anchor="w")
            hint.pack(fill="x", padx=4)
            self._rows.append(hint)
            return

        for entry in entries:
            row = ctk.CTkFrame(self.list_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            start, end = entry.span()
            when = f"{start:.2f}s" if entry.is_interval else f"@ {start:.2f}s"
            if entry.is_interval:
                when = f"{start:.2f}–{end:.2f}s"
            button = ctk.CTkButton(
                row, text=f"{when}  {entry.label}", anchor="w",
                fg_color="transparent", hover_color="#3a3a3a", height=22,
                font=ctk.CTkFont(size=11),
                command=lambda t=start: self.actions.jump_to(t))
            button.pack(side="left", fill="x", expand=True)
            delete = ctk.CTkButton(
                row, text="✕", width=24, height=22, fg_color="#8B3A3A",
                hover_color="#6E2E2E",
                command=lambda lid=entry.label_id: self._delete(lid))
            delete.pack(side="right")
            self._rows.extend([row, button, delete, ])

    def _delete(self, label_id: str) -> None:
        if self.repository.remove(label_id):
            self.refresh()
            self.actions.on_change()
