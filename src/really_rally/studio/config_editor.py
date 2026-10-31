"""Configuration tuning console.

Two tiers, and the distinction is the point:

- **Replayable** parameters are recomputed from cached features. Changing them
  and pressing Apply re-runs detection in milliseconds — no video, no MediaPipe.
- **Requires re-analysis** parameters change feature extraction itself. A pose
  model or frame-sampling change cannot be replayed from a feature store, and
  pretending otherwise would produce a confident answer that is simply wrong.
  These are displayed, greyed, and labelled rather than hidden.

Sensible defaults only: the panel exposes the parameters a person actually
tunes when a rally is missed or invented, not all 24 replayable constants.

:func:`parse_override_values` is pure so the parsing and diffing can be tested
without a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk

from really_rally.core.config_snapshot import REANALYSIS_KEYS, ConfigSnapshot


@dataclass(frozen=True)
class ParameterSpec:
    name: str          # the config.py constant name
    label: str         # what a person calls it
    hint: str = ""


# Tuning surface: the thresholds and gaps that decide whether a stroke is
# accepted and how strokes join into a rally.
REPLAYABLE_PARAMETERS: List[ParameterSpec] = [
    ParameterSpec("ARM_ENERGY_K_MULTIPLIER", "Arm energy K",
                  "Higher = arm energy must stand further above the local baseline"),
    ParameterSpec("WHIP_RATIO_K_MULTIPLIER", "Whip ratio K",
                  "Higher = whip must stand further above the local baseline"),
    ParameterSpec("ADAPTIVE_WINDOW_SEC", "Adaptive window (s)",
                  "How far back the resting baseline looks"),
    ParameterSpec("MIN_ENERGY_FLOOR", "Min energy floor", "Floor on the arm threshold"),
    ParameterSpec("MIN_WHIP_FLOOR", "Min whip floor", "Floor on the whip threshold"),
    ParameterSpec("COARSE_CANDIDATE_THRESHOLD", "Candidate threshold",
                  "Total evidence needed to call a frame a candidate"),
    ParameterSpec("COARSE_W_ARM", "Weight: arm energy", ""),
    ParameterSpec("COARSE_W_WHIP", "Weight: whip ratio", ""),
    ParameterSpec("COARSE_W_HIP", "Weight: hip velocity", ""),
    ParameterSpec("COARSE_W_SPATIAL", "Weight: court coverage", ""),
    ParameterSpec("COARSE_MIN_ABS_ARM", "Min absolute arm", "Raw-space gate on the Z-score"),
    ParameterSpec("COARSE_MIN_ABS_WHIP", "Min absolute whip", "Raw-space gate on the Z-score"),
    ParameterSpec("RALLY_START_GAP_SEC", "Rally start gap (s)",
                  "Two candidates within this confirm a rally"),
    ParameterSpec("RALLY_CONTINUE_GAP_SEC", "Rally continue gap (s)",
                  "A gap longer than this ends a rally"),
    ParameterSpec("MIN_RALLY_DURATION_SEC", "Min rally duration (s)", ""),
    ParameterSpec("BOUNDARY_PADDING_SEC", "Boundary padding (s)", ""),
    ParameterSpec("HIP_VELOCITY_STROKE_THRESHOLD", "Hip velocity min", ""),
    ParameterSpec("SPATIAL_VARIANCE_STROKE_MIN", "Spatial variance min", ""),
    ParameterSpec("PLAYER_HEIGHT_MIN", "Player height min", "Rejects distant poses"),
    ParameterSpec("PLAYER_HEIGHT_MAX", "Player height max", "Rejects implausibly large poses"),
]

# Shown so the boundary is visible, not hidden. These cannot be applied from the
# panel because their inputs are not in the feature store.
REANALYSIS_LABELS = {
    "MP_MODEL_COMPLEXITY": "Pose model complexity",
    "PROCESS_EVERY_N_FRAMES": "Frame sampling stride",
    "MP_MIN_DETECTION_CONF": "Pose detection confidence",
    "MP_MIN_TRACKING_CONF": "Pose tracking confidence",
    "GAP_MERGE_THRESHOLD_SEC": "Motion gap merge",
    "MIN_MOTION_SEGMENT_SEC": "Min motion segment",
    "MOTION_SCAN_SKIP_FRAMES": "Motion scan stride",
    "PIXEL_DIFF_THRESHOLD": "Pixel diff threshold",
    "PIXEL_CHANGE_THRESHOLD": "Pixel change threshold",
    "POSE_DROPOUT_BRIDGE_STEPS": "Pose dropout bridging",
    "MOVEMENT_WINDOW_SEC": "Movement window",
}


def parse_override_values(
    text_values: Dict[str, str],
    baseline: ConfigSnapshot,
) -> Tuple[Dict[str, float], List[str]]:
    """Turn edited text fields into a validated override dict.

    Returns ``(overrides, errors)``. Only fields that differ from ``baseline``
    become overrides, so pressing Apply without editing anything is a no-op
    rather than a config change. Blank fields are ignored; unparseable ones are
    reported and excluded, so a single typo cannot silently discard the rest of
    the edit.
    """
    overrides: Dict[str, float] = {}
    errors: List[str] = []

    for name, raw in text_values.items():
        text = (raw or "").strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            errors.append(f"{name}: {text!r} is not a number")
            continue

        field = name.lower()
        if not hasattr(baseline, field):
            errors.append(f"{name}: unknown parameter")
            continue
        if value != getattr(baseline, field):
            overrides[name] = value

    return overrides, errors


class ConfigEditor:
    """Editable replayable parameters plus a read-only re-analysis list."""

    def __init__(self, parent, on_apply, on_reset):
        self.on_apply = on_apply
        self.on_reset = on_reset
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._entries: Dict[str, ctk.CTkEntry] = {}
        self._baseline: Optional[ConfigSnapshot] = None
        self._build()

    def widget(self):
        return self.frame

    def _build(self) -> None:
        ctk.CTkLabel(self.frame, text="Replayable — recomputed from cached features",
                     font=ctk.CTkFont(size=12, weight="bold"), anchor="w"
                     ).pack(fill="x", padx=8, pady=(6, 2))

        ctk.CTkLabel(
            self.frame,
            text="These change the decision, not the measurements, so Apply "
                 "re-runs detection instantly without touching the video.",
            font=ctk.CTkFont(size=10), text_color="gray60", anchor="w",
            wraplength=380, justify="left").pack(fill="x", padx=8, pady=(0, 4))

        grid = ctk.CTkScrollableFrame(self.frame, label_text="Parameters")
        grid.pack(fill="both", expand=True, padx=8, pady=4)

        for spec in REPLAYABLE_PARAMETERS:
            row = ctk.CTkFrame(grid, fg_color="transparent")
            row.pack(fill="x", pady=1)
            label = ctk.CTkLabel(row, text=spec.label, anchor="w", width=170,
                                 font=ctk.CTkFont(size=11))
            label.pack(side="left")
            entry = ctk.CTkEntry(row, width=90)
            entry.pack(side="right")
            self._entries[spec.name] = entry
            if spec.hint:
                entry.bind("<Enter>", lambda _e, h=spec.hint: self._show_hint(h))

        self.lbl_hint = ctk.CTkLabel(self.frame, text="", anchor="w",
                                     font=ctk.CTkFont(size=10), text_color="gray60",
                                     wraplength=380, justify="left")
        self.lbl_hint.pack(fill="x", padx=8)

        buttons = ctk.CTkFrame(self.frame, fg_color="transparent")
        buttons.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(buttons, text="Apply & Replay", width=130,
                      command=self._apply).pack(side="left", padx=2)
        ctk.CTkButton(buttons, text="Reset", width=80, fg_color="gray",
                      hover_color="darkgray", command=self._reset).pack(side="left", padx=2)

        self.lbl_status = ctk.CTkLabel(self.frame, text="", anchor="w",
                                       font=ctk.CTkFont(size=11))
        self.lbl_status.pack(fill="x", padx=8)

        ctk.CTkLabel(self.frame, text="Requires video re-analysis",
                     font=ctk.CTkFont(size=12, weight="bold"), anchor="w"
                     ).pack(fill="x", padx=8, pady=(10, 2))
        ctk.CTkLabel(
            self.frame,
            text="These change how features are extracted, so they cannot be "
                 "replayed from cached data. Re-run the video to apply them.",
            font=ctk.CTkFont(size=10), text_color="gray60", anchor="w",
            wraplength=380, justify="left").pack(fill="x", padx=8)

        listed = [name for name in REANALYSIS_LABELS if name in REANALYSIS_KEYS]
        ctk.CTkLabel(self.frame, text="\n".join(f"· {REANALYSIS_LABELS[n]}" for n in listed),
                     anchor="w", justify="left", text_color="gray50",
                     font=ctk.CTkFont(size=10)).pack(fill="x", padx=16, pady=(2, 8))

    def _show_hint(self, hint: str) -> None:
        self.lbl_hint.configure(text=hint)

    # -- state ------------------------------------------------------------

    def load(self, snapshot: ConfigSnapshot) -> None:
        """Populate the fields from a configuration snapshot."""
        self._baseline = snapshot
        for name, entry in self._entries.items():
            value = getattr(snapshot, name.lower())
            entry.delete(0, "end")
            entry.insert(0, f"{value:g}")

    def current_values(self) -> Dict[str, str]:
        return {name: entry.get() for name, entry in self._entries.items()}

    def _apply(self) -> None:
        if self._baseline is None:
            return
        overrides, errors = parse_override_values(self.current_values(), self._baseline)
        if errors:
            self.lbl_status.configure(text="; ".join(errors), text_color="#ff6b6b")
            return
        self.on_apply(overrides)

    def _reset(self) -> None:
        if self._baseline is not None:
            self.load(self._baseline)
        self.lbl_status.configure(text="")
        self.on_reset()

    def set_status(self, text: str, error: bool = False) -> None:
        self.lbl_status.configure(text=text, text_color="#ff6b6b" if error else "gray80")
