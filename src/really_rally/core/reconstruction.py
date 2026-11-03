"""Reconstruct a selected stroke at source frame rate with an isolated pose tracker."""
import base64
import json
import hashlib
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import cv2
import mediapipe as mp
import numpy as np

from really_rally.core.arm_flow import arm_flow
from really_rally.core.continuity import complete_pose, STATUS, FORWARD, TEMPORAL
from really_rally.core.config import MP_MODEL_COMPLEXITY, MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF
from really_rally.core.floor import floor_track, sole_heights
from really_rally.core.motion_model import write_model, LIMBS
from really_rally.core.stroke_analysis import fit_body, measurements, write_measurements
from really_rally.utils.media_io import probe_video
from really_rally.utils.subprocesses import check_cancelled

VISIBILITY = 0.5


def stabilize(world, visibility, times=None):
    """Small centered filter on visible samples only. Never fill a missing joint."""
    valid = (visibility >= VISIBILITY) & np.isfinite(world).all(axis=2)
    visible = np.where(valid[..., None], world, 0)
    weighted = 2 * visible
    weights = 2 * valid.astype(float)
    adjacent = np.ones(len(world)-1, dtype=bool)
    if times is not None and len(times) > 1:
        gaps = np.diff(times)
        adjacent = (gaps > 0) & (gaps <= min(.10, 2.5 * np.median(gaps)))
    weighted[1:] += visible[:-1] * adjacent[:, None, None]
    weights[1:] += valid[:-1] * adjacent[:, None]
    weighted[:-1] += visible[1:] * adjacent[:, None, None]
    weights[:-1] += valid[1:] * adjacent[:, None]
    return np.where(valid[..., None], weighted / np.maximum(weights[..., None], 1), np.nan)


def _landmarks(result, box, width, height):
    image = np.full((33, 3), np.nan, np.float32)
    world = image.copy()
    visibility = np.zeros(33, np.float32)
    if result.pose_landmarks and result.pose_world_landmarks:
        image[:] = [(p.x, p.y, p.z) for p in result.pose_landmarks.landmark]
        world[:] = [(p.x, p.y, p.z) for p in result.pose_world_landmarks.landmark]
        visibility[:] = [p.visibility for p in result.pose_landmarks.landmark]
        left, top, right, bottom = box
        image[:, 0] = (left + image[:, 0] * (right - left)) / width
        image[:, 1] = (top + image[:, 1] * (bottom - top)) / height
        image[:, 2] *= (right - left) / width
    return image, world, visibility


def _hand_image(result, box, width, height):
    """Hand depth is wrist-relative in image units, never body-world depth."""
    hands = np.full((2, 21, 3), np.nan, np.float32)
    left, top, right, bottom = box
    for side, hand in enumerate((result.left_hand_landmarks, result.right_hand_landmarks)):
        if hand:
            hands[side] = [(p.x, p.y, p.z) for p in hand.landmark]
            hands[side, :, 0] = (left + hands[side, :, 0] * (right-left)) / width
            hands[side, :, 1] = (top + hands[side, :, 1] * (bottom-top)) / height
            hands[side, :, 2] *= (right-left) / width
    return hands


def _wrist_observations(image, hands, visibility, size):
    """Accept corroborating hand wrists only near an unambiguous same-side pose wrist."""
    pixels = image[..., :2].astype(float) * size
    pixels[visibility < VISIBILITY] = np.nan
    fused = pixels.copy()
    accepted = np.zeros((len(image), 2), dtype=bool)
    torso_px = np.linalg.norm(pixels[:, [11, 12]].mean(axis=1) - pixels[:, [23, 24]].mean(axis=1), axis=1)
    for side, wrist, other in ((0, 15, 16), (1, 16, 15)):
        palm = hands[:, side, 0, :2] * size
        gap = np.linalg.norm(palm - pixels[:, wrist], axis=1)
        other_gap = np.linalg.norm(palm - pixels[:, other], axis=1)
        ok = ((gap <= .20 * torso_px) & (~np.isfinite(other_gap) | (gap + .05 * torso_px < other_gap))
              & np.isfinite(hands[:, side]).all(axis=(1, 2)))
        accepted[:, side] = ok
        fused[ok, wrist] = (pixels[ok, wrist] + palm[ok]) / 2
    return fused, accepted


def _read_window(video, start, end, fps, roi, impact, stop_event=None):
    capture = cv2.VideoCapture(str(video))
    images, worlds, visibilities, previews, times = [], [], [], [], []
    hands = []
    tracker = None
    nominal_time = False
    try:
        tracker = mp.solutions.holistic.Holistic(model_complexity=MP_MODEL_COMPLEXITY,
                                                 min_detection_confidence=MP_MIN_DETECTION_CONF,
                                                 min_tracking_confidence=MP_MIN_TRACKING_CONF)
        if not capture.isOpened() or (start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start)):
            raise ValueError("Could not seek to the selected stroke.")
        for index in range(start, end + 1):
            check_cancelled(stop_event)
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            box = np.rint(np.asarray(roi) * [width, height, width, height]).astype(int)
            left, top, right, bottom = box
            if right - left < 8 or bottom - top < 8:
                raise ValueError("Player region is too small. Include the whole moving player.")
            crop = frame[top:bottom, left:right]
            result = tracker.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image, world, visibility = _landmarks(result, box, width, height)
            hands.append(_hand_image(result, box, width, height))
            images.append(image)
            worlds.append(world)
            visibilities.append(visibility)
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(timestamp) or (index > 0 and timestamp <= 0) or (times and timestamp <= times[-1]):
                nominal_time = True
            times.append(timestamp)
            ratio = min(1, 640 / max(height, width))
            preview = cv2.resize(frame, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 74])
            if not ok:
                raise ValueError(f"Could not encode preview for frame {index}.")
            previews.append(base64.b64encode(jpeg).decode("ascii"))
    finally:
        capture.release()
        if tracker is not None:
            tracker.close()
    if start + len(images) <= impact:
        raise ValueError("Could not decode the selected impact frame.")
    frames = np.arange(start, start + len(images))
    if nominal_time:
        times = frames / fps
    return (np.array(images), np.array(worlds), np.array(visibilities), frames,
            np.array(times), previews, nominal_time, [width, height], np.array(hands))


def _reverse_pose(video, frames, roi, stop_event=None):
    """Track from the other end of the stroke, revisiting source pixels in bounded blocks."""
    images = np.full((len(frames), 33, 3), np.nan, np.float32)
    worlds, confidence = images.copy(), np.zeros(images.shape[:2], np.float32)
    gray_frames = [None] * len(frames)
    capture = cv2.VideoCapture(str(video))
    try:
        with mp.solutions.pose.Pose(model_complexity=MP_MODEL_COMPLEXITY, smooth_landmarks=False,
                                    min_detection_confidence=MP_MIN_DETECTION_CONF,
                                    min_tracking_confidence=MP_MIN_TRACKING_CONF) as tracker:
            for stop in range(len(frames), 0, -12):
                check_cancelled(stop_event)
                start = max(0, stop - 12)
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frames[start])):
                    break
                block = []
                for i in range(start, stop):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    h, w = frame.shape[:2]
                    box = np.rint(np.asarray(roi) * [w, h, w, h]).astype(int)
                    x, y, x1, y1 = box
                    crop = frame[y:y1, x:x1]
                    ratio = min(1, 768 / max(crop.shape[:2]))
                    if ratio < 1:
                        crop = cv2.resize(crop, (max(1, round((x1-x)*ratio)), max(1, round((y1-y)*ratio))),
                                          interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    ok, jpeg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if ok:
                        gray_frames[i] = jpeg
                    block.append((i, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), box, w, h))
                if len(block) != stop-start:
                    break
                for i, rgb, box, w, h in reversed(block):
                    check_cancelled(stop_event)
                    images[i], worlds[i], confidence[i] = _landmarks(tracker.process(rgb), box, w, h)
    finally:
        capture.release()
    return images, worlds, confidence, gray_frames


def reconstruct(video, output, impact, stroke="stroke", before=1.0, after=1.0, roi=(0, 0, 1, 1), footwork="supported",
                *, anchor_kind="manual", hitting_hand="unknown", stop_event=None, portable_html=False):
    """Publish one portable replay, animated GLB and raw pose archive per run."""
    check_cancelled(stop_event)
    if hitting_hand not in ("left", "right", "unknown"):
        raise ValueError("Hitting hand must be left, right, or unknown.")
    if anchor_kind not in ("manual", "serve_contact", "motion_peak", "clip_midpoint"):
        raise ValueError("Unrecognized replay reference frame type.")
    if footwork not in ("supported", "unconstrained"):
        raise ValueError("Footwork must be supported or unconstrained.")
    video = Path(video).resolve()
    meta = probe_video(str(video))
    fps = meta["fps"]
    if not 0 <= impact < meta["frame_count"]:
        raise ValueError(f"Impact must be a source frame from 0 to {meta['frame_count'] - 1}.")
    if not (np.isfinite([before, after]).all() and min(before, after) >= 0 and 0 < before + after <= 10):
        raise ValueError("Select a nonempty stroke window of at most 10 seconds.")
    if not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("Use a normalized player box: left top right bottom, each from 0 to 1.")
    start = max(0, impact - round(before * fps))
    end = min(meta["frame_count"] - 1, impact + round(after * fps))
    image, world, visibility, frames, times, previews, nominal_time, size, hands = _read_window(
        video, start, end, fps, roi, impact, stop_event)
    filtered = stabilize(world, visibility, times)
    # MediaPipe to a right-handed, Y-up display with the camera on positive Z.
    pixels, accepted = _wrist_observations(image, hands, visibility, size)
    fitted, scale, offset, reference = fit_body(filtered, pixels, visibility)
    display = fitted * [1, -1, -1]
    readings = measurements(display, filtered, reference, frames, times, impact - start, pixels)
    visible = np.isfinite(display).all(axis=2)
    drawable = np.any([visible[:, a] & visible[:, b] for _, a, b, _ in LIMBS], axis=0)
    if not drawable.any():
        raise ValueError("No visible body segments in this window. Check the player box and recording.")
    reverse_image, reverse_world, reverse_vis, gray_frames = _reverse_pose(video, frames, roi, stop_event)
    reverse_filtered = stabilize(reverse_world, reverse_vis, times)
    reverse_pixels = reverse_image[..., :2] * size
    reverse_pixels[reverse_vis < VISIBILITY] = np.nan
    reverse_fit, _, _, _ = fit_body(reverse_filtered, reverse_pixels, reverse_vis)
    crop_box = np.rint(np.asarray(roi) * [*size, *size]).astype(int)
    flow = arm_flow(gray_frames, crop_box, size, times, (image, visibility), (reverse_image, reverse_vis), hands, stop_event=stop_event)
    del gray_frames
    check_cancelled(stop_event)
    trajectory, quality = complete_pose(fitted, filtered, reference, image, visibility, times, size,
                                        (reverse_fit, reverse_filtered, reverse_image, reverse_vis), flow, (scale, offset))
    completed = trajectory * [1, -1, -1]
    if footwork == "supported":
        root_y, support = floor_track(display, pixels, size, times)
        # Only original shoe observations can establish support. Keep that foot
        # fixed and prevent a reconstructed free shoe from passing through it.
        for side in (0,1):
            rows = np.flatnonzero(support == side)
            ids = np.array([27,29,31])+side
            completed[np.ix_(rows,ids)] = display[np.ix_(rows,ids)]
            quality[np.ix_(rows,ids)] = FORWARD
        clearance = sole_heights(completed) + root_y[:,None]
        for side in (0,1):
            ids = np.array([27,29,31])+side
            rows = np.flatnonzero((support >= 0) & (support != side) & (clearance[:,side] < -1e-6)
                                  & (quality[:,ids] > FORWARD).any(axis=1))
            completed[np.ix_(rows,ids,[1])] -= clearance[rows,side,None,None]
            quality[np.ix_(rows,ids)] = TEMPORAL
        trajectory = completed * [1,-1,-1]
    else:
        root_y, support = np.zeros(len(times)), np.full(len(times), -1)
    projected_track = trajectory[..., :2]*scale[:, None, None] + offset[:, None]
    curves = measurements(completed, completed, reference, frames, times, impact-start, projected_track)
    info = dict(version=6, anchor_kind=anchor_kind, hitting_hand=hitting_hand, joint_status=STATUS, joint_status_counts=np.bincount(quality.ravel(), minlength=len(STATUS)).tolist(), footwork=footwork, floor_anchored_frames=int((support >= 0).sum()), surface="Articulated neutral mannequin posed from fitted landmarks.", source=str(video), source_size=size, fps=fps, impact_frame=int(impact),
                pose_model="MediaPipe Holistic", hand_order=["left", "right"],
                hand_coordinates="Normalized source image x/y, wrist-relative image z. Not registered to body xyz.",
                fit="Weak-perspective image fit with soft median link lengths; 0.09 m correction cap.",
                stroke=stroke, player_box=list(roi), mediapipe=mp.__version__, model_complexity=MP_MODEL_COMPLEXITY,
                coordinates="Raw world in estimated meters, hip origin. Display uses (x, -y, -z) plus display_root_y on Y. Supported soles meet Y=0.",
                image_coordinates="Normalized to the full decoded source frame.",
                visibility_cutoff=VISIBILITY, timestamp_basis="frame/fps" if nominal_time else "decoder",
                drawable_frames=int(drawable.sum()), total_frames=len(frames))
    check_cancelled(stop_event)
    parent = Path(output)
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stroke-", dir=parent))
    final = parent / f"{video.stem}-f{impact}-{stage.name[8:]}"
    try:
        np.savez_compressed(stage / "pose.npz", frame=frames, timestamp=times, image=image,
                            world=world, visibility=visibility, filtered_world=filtered, fitted_world=fitted,
                            hand_image=hands, hand_accepted=accepted, fit_pixels=pixels,
                            pixels_per_meter=scale, pixel_offset=offset, display_root_y=root_y, support_foot=support,
                            trajectory_world=trajectory, trajectory_status=quality, reverse_image=reverse_image,
                            reverse_world=reverse_world, reverse_visibility=reverse_vis, arm_flow_pixels=flow)
        write_measurements(stage / "kinematics.csv", readings)
        (stage / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        export_rows = [dict(row, **{f"joint_{j}_source": STATUS[int(q)] for j,q in enumerate(mask)})
                       for row,mask in zip(curves, quality)]
        write_measurements(stage / "trajectory.csv", export_rows)
        write_model(stage / "player.glb", completed, times - times[0], root_y=root_y)
        # The desktop clip is self-contained and frame-addressable without a browser.
        with ZipFile(stage / "frames.zip", "w") as archive:
            for frame, preview in zip(frames, previews):
                check_cancelled(stop_event)
                archive.writestr(f"{frame}.jpg", base64.b64decode(preview))
        if portable_html:
            # JSON null explicitly denotes unavailable joints in the browser.
            joints = [[point.round(5).tolist() if np.isfinite(point).all() else None for point in row]
                      for row in completed]
            overlay = [[xy[:2].round(5).tolist() if valid and np.isfinite(xy[:2]).all() else None
                        for xy, valid in zip(row, mask)] for row, mask in zip(image, visible)]
            hand_overlay = [[hand[:, :2].round(5).tolist() if ok else None for hand, ok in zip(row, mask)]
                            for row, mask in zip(hands, accepted)]
            projected = (fitted[..., :2] * scale[:, None, None] + offset[:, None]) / size
            fit_overlay = [[xy.round(5).tolist() if np.isfinite(xy).all() else None for xy in row]
                           for row in projected]
            projected_track /= size
            track_overlay = [[xy.round(5).tolist() if np.isfinite(xy).all() else None for xy in row]
                             for row in projected_track]
            stat = video.stat()
            source_event = json.dumps([str(video), stat.st_size, stat.st_mtime_ns, impact])
            shot_id = hashlib.sha256(source_event.encode()).hexdigest()[:24]
            identity = json.dumps([shot_id, list(roi), stroke, int(frames[0]), int(frames[-1])])
            review_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
            replay = dict(anchor_kind=anchor_kind, hitting_hand=hitting_hand, shot_id=shot_id, review_id=review_id, fit_overlay=fit_overlay, kinematics=readings, hand_overlay=hand_overlay, stroke=stroke, impact=impact, frames=frames.tolist(), times=times.tolist(),
                          image_size=size, overlay=overlay, track_overlay=track_overlay, curves=curves, joint_status=quality.tolist(),
                          images=previews, joints=joints, limbs=LIMBS, footwork=footwork,
                          root_y=root_y.tolist(), support=support.tolist(),
                          mannequin=base64.b64encode((stage / "player.glb").read_bytes()).decode("ascii"))
            template = Path(__file__).with_name("stroke_replay.html").read_text(encoding="utf-8")
            review_script = Path(__file__).with_name("shot_review.js").read_text(encoding="utf-8")
            template = template.replace("__SHOT_REVIEW_SCRIPT__", review_script)
            renderer = Path(__file__).with_name("mannequin.js").read_text(encoding="utf-8")
            template = template.replace("__MANNEQUIN_SCRIPT__", renderer)
            payload = json.dumps(replay, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
            (stage / "replay.html").write_text(template.replace("__REPLAY_DATA__", payload), encoding="utf-8")
        check_cancelled(stop_event)
        stage.rename(final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return final.resolve()
