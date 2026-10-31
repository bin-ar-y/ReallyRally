"""Debug + Label Studio — a time-synchronized debugging and annotation workspace.

The studio is deliberately layered so detection stays testable without a GUI:

- :mod:`really_rally.core.detection` / :mod:`really_rally.utils.feature_store`
  hold the pure detection and persistence logic (no Tk, no OpenCV).
- the repositories here (:mod:`telemetry`, :mod:`annotation`, :mod:`replay`)
  load and transform data and never touch widgets.
- :mod:`app` and :mod:`controller` own the widgets and the worker thread.

Nothing in this package runs detection inside a Tk callback; the GUI renders
what the worker produced.
"""

__all__ = ["app", "controller", "telemetry", "video_navigator", "frame_inspector"]
