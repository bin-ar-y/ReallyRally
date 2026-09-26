"""Current-frame inspector: what the detector saw and what it decided.

A value on its own is not interpretable. ``0.023`` is meaningless until you know
it is a raw arm-energy measurement, that the bar it had to clear was ``0.249``,
and that it therefore did *not* fire. This module renders each frame as
RAW / THRESHOLD / DERIVED SCORE / DECISION / STATE so the interpretation travels
with the number.

:func:`build_inspector` is pure and returns plain dataclasses; the Tk widget
only draws what it returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from really_rally.core.config import PLAYER_HEIGHT_MAX, PLAYER_HEIGHT_MIN
from really_rally.utils.feature_store import (
    FEATURE_ROLES,
    ROLE_CONTEXT,
    ROLE_DECISION,
    ROLE_POSE,
    ROLE_RAW,
    ROLE_SCORE,
    ROLE_STATE,
    ROLE_THRESHOLD,
)

# Which raw signal each coarse-detector evidence name corresponds to. Mirrors
# CoarseCandidateDetector.score(), which appends exactly these names. A signal
# missing from this table still displays its value; only the trigger line is
# skipped.
_FIRE_NAMES = {
    "arm_energy": "arm", "smoothed_arm": "arm",
    "whip_ratio": "whip", "smoothed_whip": "whip",
    "hip_vel": "hip",
    "spatial_variance": "spatial",
}

# Signals whose raw/smoothed/threshold/z names do not follow a mechanical
# pattern (t_arm pairs with arm_energy, not with "arm"). Everything else is
# derived from the key prefix, so a new t_<x> column still gets its own group.
_SIGNAL_ALIASES = {
    "arm": ("Arm Energy", "arm_energy"),
    "whip": ("Whip Ratio", "whip_ratio"),
}


@dataclass
class InspectorRow:
    label: str
    value: str
    role: str
    emphasis: bool = False


@dataclass
class InspectorSection:
    title: str
    rows: List[InspectorRow] = field(default_factory=list)


def _format(key: str, value) -> str:
    if key in ("is_valid_pose", "candidate"):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return value or "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key == "frame":
        return str(int(number))
    return f"{number:.4f}"


def _present(record: Dict, key: str) -> bool:
    return key in record and record[key] not in ("", None)


def _row(record: Dict, key: str, label: str, role: str, emphasis: bool = False) -> Optional[InspectorRow]:
    if not _present(record, key):
        return None
    return InspectorRow(label, _format(key, record[key]), role, emphasis)


def build_inspector(record: Optional[Dict]) -> List[InspectorSection]:
    """Render one feature record as interpretable, grouped rows."""
    if not record:
        return [InspectorSection("Frame", [InspectorRow("status", "no telemetry loaded", ROLE_CONTEXT)])]

    fired = {name for name in str(record.get("fired_signals", "")).split("|") if name}
    sections: List[InspectorSection] = []
    rendered = set()

    # -- context ---------------------------------------------------------
    context = [r for r in (
        _row(record, "frame", "Frame", ROLE_CONTEXT),
        _row(record, "segment", "Segment", ROLE_CONTEXT),
    ) if r]
    if _present(record, "time"):
        context.insert(1, InspectorRow("Time", f"{float(record['time']):.2f}s", ROLE_CONTEXT))
    rendered.update({"frame", "time", "segment"})
    sections.append(InspectorSection("Frame", context))

    # -- one section per detected/thresholded signal ---------------------
    for threshold_key in sorted(k for k in record if k.startswith("t_")):
        stem = threshold_key[2:]
        title, raw_key = _SIGNAL_ALIASES.get(stem, (stem.replace("_", " ").title(), None))
        rows: List[InspectorRow] = []
        if raw_key:
            raw_row = _row(record, raw_key, "Raw", ROLE_RAW)
            if raw_row:
                rows.append(raw_row)
        smoothed_row = _row(record, f"smoothed_{stem}", "Smoothed", ROLE_RAW)
        if smoothed_row:
            rows.append(smoothed_row)
        threshold_row = _row(record, threshold_key, "Threshold", ROLE_THRESHOLD)
        if threshold_row:
            rows.append(threshold_row)
        z_row = _row(record, f"z_{stem}", "Z-score", ROLE_SCORE)
        if z_row:
            rows.append(z_row)

        if z_row and threshold_row:
            # The decision this pair produces, stated explicitly.
            try:
                triggered = float(record.get(f"z_{stem}", 0.0)) > float(record[threshold_key])
            except (TypeError, ValueError):
                triggered = False
            rows.append(InspectorRow("Trigger", "TRUE" if triggered else "FALSE",
                                     ROLE_DECISION, emphasis=True))
        elif raw_key in _FIRE_NAMES:
            fire_name = _FIRE_NAMES[raw_key]
            rows.append(InspectorRow("Evidence fired",
                                     "TRUE" if fire_name in fired else "FALSE",
                                     ROLE_DECISION, emphasis=True))
        # A threshold key is only meaningful together with its threshold.
        if raw_key:
            rendered.add(raw_key)
        rendered.update({f"smoothed_{stem}", threshold_key, f"z_{stem}"})
        if rows:
            sections.append(InspectorSection(title, rows))

    # -- remaining signals that have no threshold pairing ----------------
    paired = set()
    for threshold_key in (k for k in record if k.startswith("t_")):
        stem = threshold_key[2:]
        _, raw_key = _SIGNAL_ALIASES.get(stem, (stem, None))
        paired.update({f"smoothed_{stem}", threshold_key, f"z_{stem}"})
        if raw_key:
            paired.add(raw_key)
    others = [r for r in (
        _row(record, key, key.replace("_", " ").title(), ROLE_RAW)
        for key in ("energy", "hip_vel", "spatial_variance", "action_score", "player_height")
        if key not in paired
    ) if r]
    rendered.update({"energy", "hip_vel", "spatial_variance", "action_score", "player_height"})
    if others:
        sections.append(InspectorSection("Other signals", others))

    # -- score / decision ------------------------------------------------
    scores = [r for r in (
        _row(record, "coarse_score", "Coarse score", ROLE_SCORE, emphasis=True),
    ) if r]
    rendered.add("coarse_score")
    if scores:
        sections.append(InspectorSection("Derived score", scores))

    decision = [r for r in (
        _row(record, "is_valid_pose", "Pose valid", ROLE_DECISION),
        _row(record, "candidate", "Candidate", ROLE_DECISION, emphasis=True),
    ) if r]
    if _present(record, "player_height"):
        height = float(record["player_height"])
        in_range = PLAYER_HEIGHT_MIN <= height <= PLAYER_HEIGHT_MAX
        decision.append(InspectorRow(
            "Distance gate",
            f"TRUE ({height:.3f} in {PLAYER_HEIGHT_MIN}–{PLAYER_HEIGHT_MAX})" if in_range
            else f"FALSE ({height:.3f} outside {PLAYER_HEIGHT_MIN}–{PLAYER_HEIGHT_MAX})",
            ROLE_DECISION, emphasis=not in_range))
    # "no evidence fired" is itself the answer to "why was this rejected", so
    # the row is shown even when the value is empty.
    if "fired_signals" in record:
        decision.append(InspectorRow("Fired evidence",
                                     str(record["fired_signals"]) or "none", ROLE_DECISION))
    rendered.update({"is_valid_pose", "candidate", "fired_signals", "player_height"})
    if decision:
        sections.append(InspectorSection("Decision", decision))

    if _present(record, "state"):
        sections.append(InspectorSection("State", [
            InspectorRow("Detector state", str(record["state"]), ROLE_STATE, emphasis=True)]))
    rendered.add("state")

    # -- everything else, so no column is ever silently dropped -----------
    # Pose geometry first (long and least often read), then any column the
    # groups above did not claim. The catch-all is what lets racket/ball/fused
    # telemetry appear the moment it is written, with no change here.
    remainder = [k for k in record if k not in rendered and _present(record, k)]
    pose_rows = [InspectorRow(k, _format(k, record[k]), ROLE_POSE)
                 for k in remainder if FEATURE_ROLES.get(k) == ROLE_POSE]
    unknown_rows = [InspectorRow(k, _format(k, record[k]), ROLE_RAW)
                    for k in remainder if FEATURE_ROLES.get(k) != ROLE_POSE]
    if pose_rows:
        sections.append(InspectorSection("Pose geometry", pose_rows))
    if unknown_rows:
        sections.append(InspectorSection("Other telemetry", unknown_rows))

    return sections


class FrameInspector:
    """Renders :func:`build_inspector` output into a scrollable panel."""

    def __init__(self, parent):
        import customtkinter as ctk

        self.frame = ctk.CTkScrollableFrame(parent, label_text="Current frame")
        self._widgets: List = []
        self._ctk = ctk

    def widget(self):
        return self.frame

    def update(self, record: Optional[Dict]) -> None:
        ctk = self._ctk
        for widget in self._widgets:
            widget.destroy()
        self._widgets = []

        for section in build_inspector(record):
            header = ctk.CTkLabel(self.frame, text=section.title,
                                  font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
            header.pack(fill="x", padx=6, pady=(8, 2))
            self._widgets.append(header)

            for row in section.rows:
                line = ctk.CTkFrame(self.frame, fg_color="transparent")
                line.pack(fill="x", padx=10)
                self._widgets.append(line)
                name = ctk.CTkLabel(line, text=row.label, anchor="w",
                                    font=ctk.CTkFont(size=11), text_color="gray70")
                name.pack(side="left")
                value = ctk.CTkLabel(
                    line, text=row.value, anchor="e",
                    font=ctk.CTkFont(size=11, weight="bold" if row.emphasis else "normal"))
                value.pack(side="right")
                self._widgets.append(name)
                self._widgets.append(value)
