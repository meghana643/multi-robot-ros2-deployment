#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


VIDEO_PATH = Path(
    "/home/smartubuntu3/Desktop/camera_test/Task1/"
    "rosbag2_2026_07_15-13_49_46/"
    "rosbag2_2026_07_15-13_49_46__topview_raw.mp4"
)

MODEL_PATH = Path(
    "/home/smartubuntu3/Desktop/camera_test/yolo_model.pt"
)

OUTPUT_PATH = Path(
    "/home/smartubuntu3/Desktop/camera_test/"
    "yolo_annotated_output.mp4"
)

CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Set to True to show the live OpenCV window.
# Set to False if the window hangs or you are running remotely.
SHOW_WINDOW = True

# Process every frame.
# Change to 2 to process every second frame, 3 for every third frame, etc.
FRAME_SKIP = 1

WINDOW_NAME = "YOLO Detection"


def check_file(path: Path, description: str) -> None:
    if not path.exists():
        print(
            f"ERROR: {description} was not found:\n{path}",
            flush=True,
        )
        sys.exit(1)

    if not path.is_file():
        print(
            f"ERROR: {description} is not a file:\n{path}",
            flush=True,
        )
        sys.exit(1)


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v is widely supported by OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    return writer


def main() -> None:
    print("=" * 60, flush=True)
    print("YOLO video detection", flush=True)
    print("=" * 60, flush=True)

    print(f"Python executable: {sys.executable}", flush=True)
    print(f"OpenCV version: {cv2.__version__}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"DISPLAY: {os.environ.get('DISPLAY')}", flush=True)

    check_file(VIDEO_PATH, "Input video")
    check_file(MODEL_PATH, "YOLO model")

    if torch.cuda.is_available():
        device = 0
        print(
            f"Using GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )
    else:
        device = "cpu"
        print("CUDA is unavailable. Using CPU.", flush=True)

    print(f"Opening video:\n{VIDEO_PATH}", flush=True)

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        print(
            "ERROR: OpenCV could not open the video.",
            flush=True,
        )
        sys.exit(1)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        print(
            "Warning: Invalid source FPS. Using 30 FPS.",
            flush=True,
        )
        fps = 30.0

    output_fps = fps / FRAME_SKIP

    print(
        f"Resolution: {width}x{height}\n"
        f"FPS: {fps:.2f}\n"
        f"Frames: {total_frames}",
        flush=True,
    )

    print(f"Loading YOLO model:\n{MODEL_PATH}", flush=True)

    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as exc:
        capture.release()
        print(
            f"ERROR: Could not load YOLO model:\n"
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        sys.exit(1)

    print("YOLO model loaded successfully.", flush=True)

    try:
        writer = create_video_writer(
            OUTPUT_PATH,
            output_fps,
            width,
            height,
        )
    except Exception as exc:
        capture.release()
        print(
            f"ERROR: {exc}",
            flush=True,
        )
        sys.exit(1)

    print(f"Output video:\n{OUTPUT_PATH}", flush=True)

    window_enabled = SHOW_WINDOW

    if window_enabled:
        try:
            print("Creating display window...", flush=True)

            cv2.namedWindow(
                WINDOW_NAME,
                cv2.WINDOW_NORMAL,
            )

            cv2.resizeWindow(
                WINDOW_NAME,
                960,
                540,
            )

            print("Display window created.", flush=True)

        except cv2.error as exc:
            print(
                "Warning: OpenCV display is unavailable.",
                flush=True,
            )
            print(exc, flush=True)
            print(
                "Continuing without a display window.",
                flush=True,
            )
            window_enabled = False

    source_frame_number = 0
    processed_frame_number = 0
    total_detections = 0
    start_total = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()

            if not ok or frame is None:
                print("End of video.", flush=True)
                break

            source_frame_number += 1

            if source_frame_number % FRAME_SKIP != 0:
                continue

            processed_frame_number += 1

            inference_start = time.perf_counter()

            try:
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    device=device,
                    verbose=False,
                )
            except Exception as exc:
                print(
                    f"\nERROR during inference on frame "
                    f"{source_frame_number}:\n"
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                break

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            result = results[0]

            try:
                annotated = result.plot()
            except Exception as exc:
                print(
                    f"\nERROR while drawing detections:\n"
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                break

            if result.boxes is None:
                detection_count = 0
            else:
                detection_count = len(result.boxes)

            total_detections += detection_count

            cv2.putText(
                annotated,
                f"Frame: {source_frame_number}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated,
                f"Detections: {detection_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated,
                f"Inference: {inference_ms:.1f} ms",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            print(
                f"\rFrame {source_frame_number}/{total_frames} | "
                f"Detections: {detection_count} | "
                f"Inference: {inference_ms:.1f} ms",
                end="",
                flush=True,
            )

            if window_enabled:
                try:
                    cv2.imshow(
                        WINDOW_NAME,
                        annotated,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        print(
                            "\nStopped by user.",
                            flush=True,
                        )
                        break

                except cv2.error as exc:
                    print(
                        "\nWarning: Display failed.",
                        flush=True,
                    )
                    print(exc, flush=True)
                    print(
                        "Continuing without the display window.",
                        flush=True,
                    )
                    window_enabled = False

    except KeyboardInterrupt:
        print(
            "\nStopped with Ctrl+C.",
            flush=True,
        )

    finally:
        capture.release()
        writer.release()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    elapsed_total = time.perf_counter() - start_total

    print("\n" + "=" * 60, flush=True)
    print("Processing finished", flush=True)
    print(
        f"Processed frames: {processed_frame_number}",
        flush=True,
    )
    print(
        f"Total detections: {total_detections}",
        flush=True,
    )
    print(
        f"Total processing time: {elapsed_total:.1f} seconds",
        flush=True,
    )
    print(
        f"Saved output video:\n{OUTPUT_PATH}",
        flush=True,
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()