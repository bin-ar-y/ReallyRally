# Labelling ground truth

Every detection number the project reports — recall, precision, F1, boundary
error — is measured against hand-labelled video. Without labels there is no way
to tell whether a change improved detection or just moved the failure around,
which is why `benchmarks/cases/` ships empty and the tooling below exists.

Budget roughly 5–10 minutes per clip once you are used to it.

---

## Why labels matter here

The project's biggest known problem is **false negatives** (missed rallies), and
the honest status right now is that **no real recall figure exists**. Published
figures so far are label-free proxies (stroke peaks in the arm-energy signal).
They are useful for comparing two versions of the code against each other, but
they cannot tell you "we find 70% of real rallies".

Labelling a handful of clips converts those proxies into real measurements.

---

## The fast workflow

### 1. Look at the clip without watching it

```bash
python benchmarks/label.py --video ~/Movies/IMG_0830.mov \
    --contact-sheet /tmp/sheet.png --frames 10
```

This writes a single PNG: ten evenly spaced frames in a row, each stamped with
its timestamp. Open it and you can immediately tell whether the clip is rally
practice, serve practice, ball feeding, or not tennis at all — and roughly where
the action sits in the timeline.

### 2. Create the benchmark case

```bash
python benchmarks/label.py --init-case serve_0830 \
    --video ~/Movies/IMG_0830.mov --mode serve
```

This creates `benchmarks/cases/serve_0830/` containing:

| File | Purpose | Tracked by git? |
|---|---|---|
| `video.mov` | copied from your source | **no** (gitignored — large) |
| `mode.txt` | `rally`, `serve` or `feed` | yes |
| `annotations.json` | your labels, starts empty | yes |

Use `--move` instead of the default copy if you don't want a duplicate file.

### 3. Label it

```bash
python benchmarks/label.py --video ~/Movies/IMG_0830.mov --case serve_0830
```

You get a prompt with the clip's duration and fps:

```
label> 12.5-25        add a segment from 12.5s to 25.0s
label> 40-55.3        add another
label> list           show everything you have so far
label> undo           remove the last one
label> sheet          write a contact sheet if you need to re-check the content
label> quit           save and exit
```

Separators are flexible: `12.5-25`, `12.5:25` and `12.5 25` all work. Input is
deliberately strict — malformed or ambiguous entries like `1-2-3` or `25-12` are
rejected rather than reinterpreted, because a silently misparsed label would
corrupt every measurement taken against it.

To label without the interactive prompt:

```bash
python benchmarks/label.py --video clip.mov --case serve_0830 \
    --add 12.5-25 --add 40-55.3
```

### 4. Check your coverage

```bash
python benchmarks/label.py --status
```

```
case                    mode    video     labels  status
----------------------------------------------------------------------
serve_0830              serve   True          12  OK
rally_0817              rally   True           0  NEEDS LABELS
```

### 5. Measure

```bash
python benchmarks/run_benchmark.py --cases benchmarks/cases \
    --output benchmarks/report.json
```

Per case and aggregated, this reports precision, **recall** (the primary metric
— missed rallies are the main failure mode), F1, missed-rally and false-rally
counts, boundary timing error in seconds, and runtime as a multiple of video
duration.

---

## What to label, and how

**Label the rally, not the stroke.** A segment runs from the first stroke to the
point where the ball is dead — the last stroke plus a short follow-through.
Don't include walking to the baseline, ball collection between rallies, or
conversation.

**Be consistent about boundaries.** The metric is temporal IoU, so a clip
labelled generously and one labelled tightly will score differently against the
same prediction. Pick a rule and apply it to every clip:

> Start: the instant of the first stroke's contact.
> End: the instant the last stroke's follow-through finishes.

**Label only what is tennis.** Ball picking, walking between points, and
shadow swings are not rallies. Noting a shadow swing as a rally will hide the
false-positive class the project is specifically trying to fix.

**Serve and feed modes have different natural units.** For `serve`, one segment
per serve (from toss to follow-through). For `feed`, one segment per fed ball.

**Add notes when something is unusual** — a shadow swing, an occluded player, a
camera pan — by editing `annotations.json` afterwards. Notes cost nothing and
are the only record of why a boundary sits where it does.

---

## Suggested first corpus

Five clips is enough to produce meaningful numbers. Cover the distinct failure
modes rather than five near-identical videos:

| # | Clip type | What it tests |
|---|---|---|
| 1 | Clean rally, steady camera | Baseline — should score well |
| 2 | Dense rally, continuous hitting | Baseline drift / adaptive threshold |
| 3 | **Shadow swing practice, no ball** | The main false-positive class |
| 4 | Ball picking / walking between points | False positives from non-rally movement |
| 5 | Serve practice, multiple serves | Serve mode + handedness locking |

Clip 3 is the most valuable one: it is the case the detector currently cannot
distinguish, and it is what the object-verification work exists to fix.

---

## Annotation format

`benchmarks/cases/<case>/annotations.json`:

```json
[
  {
    "start": 12.5,
    "end": 25.0,
    "label": "rally",
    "notes": "baseline crosscourt, ~6 strokes"
  }
]
```

`start` and `end` are seconds. Overlapping segments are merged automatically on
save.

---

## Troubleshooting

**"Not a terminal; nothing to do."** — the interactive prompt needs a TTY. Use
`--add START-END` to label non-interactively, or run it directly in your
terminal rather than through a script.

**Video not found** — `--init-case` copies the video; later `--video` runs should
point at whatever file you want to label (either the original or the copy).

**Labels disappeared** — `annotations.json` is rewritten on every save and
merged/sorted. If you hand-edit it, keep it valid JSON.
