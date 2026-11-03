"""Freeze an impact frame and choose the region occupied by the serving player."""
import tkinter as tk
from tkinter import ttk

from PIL import ImageTk


def choose_stroke(parent, image, frame):
    dialog = tk.Toplevel(parent)
    dialog.title(f"Reconstruct stroke at frame {frame}")
    dialog.transient(parent)
    dialog.grab_set()
    ttk.Label(dialog, text="Drag around the player. Include the entire swing and footwork.").pack(padx=12, pady=8)
    preview = image.copy()
    preview.thumbnail((760, 440))
    width, height = preview.size
    photo = ImageTk.PhotoImage(preview, master=dialog)
    canvas = tk.Canvas(dialog, width=width, height=height, highlightthickness=0, cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)
    rect = canvas.create_rectangle(1, 1, width-1, height-1, outline="#83ffe4", width=2)
    start = [0, 0]
    box = [0., 0., 1., 1.]
    result = []

    def drag(event):
        x, y = max(0, min(width, event.x)), max(0, min(height, event.y))
        x0, x1 = sorted((start[0], x))
        y0, y1 = sorted((start[1], y))
        canvas.coords(rect, x0, y0, x1, y1)
        box[:] = [x0/width, y0/height, x1/width, y1/height]
        build.configure(state="normal" if x1-x0 >= 8 and y1-y0 >= 8 else "disabled")

    def press(event):
        start[:] = [event.x, event.y]
        drag(event)

    def submit():
        result.append((stroke.get(), tuple(box), float(before.get()), float(after.get()),
                       "supported" if planted.get() else "unconstrained"))
        dialog.destroy()

    canvas.bind("<Button-1>", press)
    canvas.bind("<B1-Motion>", drag)
    row = ttk.Frame(dialog)
    row.pack(pady=10)
    stroke = tk.StringVar(dialog, value="forehand")
    ttk.Combobox(row, textvariable=stroke, state="readonly",
                 values=("forehand", "backhand", "serve", "other"), width=12).pack(side="left", padx=5)
    before, after = tk.StringVar(dialog, value="1"), tk.StringVar(dialog, value="1")
    for label, value in (("Seconds before", before), ("after", after)):
        ttk.Label(row, text=label).pack(side="left")
        ttk.Combobox(row, textvariable=value, state="readonly",
                     values=("0.5", "1", "2", "3", "5"), width=3).pack(side="left", padx=3)
    planted = tk.BooleanVar(dialog, value=True)
    ttk.Checkbutton(dialog, text="At least one foot stays on the court throughout this stroke", variable=planted).pack(pady=4)
    build = ttk.Button(row, text="Build 3D model", command=submit)
    build.pack(side="left")
    ttk.Button(row, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)
    parent.wait_window(dialog)
    return result[0] if result else None
