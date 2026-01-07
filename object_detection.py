from ultralytics import YOLO
import os
import glob
import cv2


def get_sorted_frame_paths(frames_dir, extension="jpg"):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, f"*.{extension}")))
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in directory: {frames_dir}")
    return frame_paths


def load_yolo_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


def run_object_detection_on_frames(
    frames_dir,
    output_dir,
    model,
    confidence=0.6,
    allowed_classes=None
):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = get_sorted_frame_paths(frames_dir)

    for path in frame_paths:
        results = model(path, conf=confidence)
        image = cv2.imread(path)

        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            conf = float(box.conf)

            if allowed_classes is not None and cls_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, image)


def frames_to_video(frames_dir, output_video_path, fps=12, extension="jpg"):
    frame_paths = get_sorted_frame_paths(frames_dir, extension)
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()


def run_object_detection_pipeline(
    frames_dir,
    detected_dir,
    output_video_path,
    model_path="yolov8n.pt",
    confidence=0.6,
    allowed_classes=None,
    fps=12
):
    model = load_yolo_model(model_path)

    run_object_detection_on_frames(
        frames_dir=frames_dir,
        output_dir=detected_dir,
        model=model,
        confidence=confidence,
        allowed_classes=allowed_classes
    )

    frames_to_video(
        frames_dir=detected_dir,
        output_video_path=output_video_path,
        fps=fps
    )
