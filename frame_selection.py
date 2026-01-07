import cv2
import numpy as np
import os
from frame_preprocessing import preprocess_frame

def select_keyframes(
    cap,
    prev_gray,
    output_dir,
    pixel_diff_thresh,
    percent_changed_thresh,
    frame_skip
):
    frame_id = 0
    saved_frames = 0
    percent_changes = []

    while True:
        for _ in range(frame_skip):
            ret, frame = cap.read()

        if not ret:
            break

        curr_gray = preprocess_frame(frame)

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_mask = cv2.threshold(
            diff,
            pixel_diff_thresh,
            255,
            cv2.THRESH_BINARY
        )

        changed_pixels = np.count_nonzero(motion_mask)
        percent_changed = (changed_pixels / motion_mask.size) * 100

        percent_changes.append(percent_changed)

        if percent_changed > percent_changed_thresh:
            filename = f"frame_{frame_id:06d}_chg_{percent_changed:.2f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_frames += 1

        prev_gray = curr_gray
        frame_id += 1

    cap.release()
    return percent_changes, saved_frames

def run_frame_selection(
    video_path,
    pixel_diff_thresh=15,
    percent_changed_thresh=0.15,
    frame_skip=3
):
    from frame_preprocessing import (
        video_preprocessing,
        preprocess_first_frame
    )

    setup = video_preprocessing(video_path)

    prev_gray = preprocess_first_frame(setup["cap"])

    percent_changes, saved_frames = select_keyframes(
        cap=setup["cap"],
        prev_gray=prev_gray,
        output_dir=setup["selected_dir"],
        pixel_diff_thresh=pixel_diff_thresh,
        percent_changed_thresh=percent_changed_thresh,
        frame_skip=frame_skip
    )

    return {
        "video_name": setup["video_name"],
        "total_frames": setup["total_frames"],
        "saved_frames": saved_frames,
        "percent_changes": percent_changes,
        "selected_dir": setup["selected_dir"]
    }
