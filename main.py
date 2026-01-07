
from frame_selection import run_frame_selection
from summarizer import summarize_frames_to_video
from object_detection import run_object_detection_on_frames
from object_detection import load_yolo_model
import os


VIDEO_PATH = "Videos/VIRAT_S_000002_sizesmall.mp4"

PIXEL_DIFF_THRESH = 15
PERCENT_CHANGED_THRESH = 0.15
FRAME_SKIP = 3

SUMMARY_FPS = 12
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.6
ALLOWED_CLASSES = {"person", "car"}


def main():
    frame_selection_result = run_frame_selection(
        video_path=VIDEO_PATH,
        pixel_diff_thresh=PIXEL_DIFF_THRESH,
        percent_changed_thresh=PERCENT_CHANGED_THRESH,
        frame_skip=FRAME_SKIP
    )

    base_dir = os.path.dirname(frame_selection_result["selected_dir"])
    selected_frames_dir = frame_selection_result["selected_dir"]
    detected_frames_dir = os.path.join(base_dir, "detected_frames")
    summaries_dir = os.path.join(base_dir, "summaries")

    os.makedirs(detected_frames_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    summarize_frames_to_video(
        frames_dir=selected_frames_dir,
        output_video_path=os.path.join(summaries_dir, "summary_raw.mp4"),
        summary_fps=SUMMARY_FPS
    )

    model = load_yolo_model(YOLO_MODEL_PATH)

    run_object_detection_on_frames(
        frames_dir=selected_frames_dir,
        output_dir=detected_frames_dir,
        model=model,
        confidence=YOLO_CONFIDENCE,
        allowed_classes=ALLOWED_CLASSES
    )

    summarize_frames_to_video(
        frames_dir=detected_frames_dir,
        output_video_path=os.path.join(summaries_dir, "summary_detected.mp4"),
        summary_fps=SUMMARY_FPS
    )


if __name__ == "__main__":
    main()
