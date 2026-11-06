"""
Real-Time Room Monitor
----------------------
Local webcam person detection using OpenCV's pretrained HOG + SVM detector.

Controls:
    Q or ESC : Quit

Privacy:
    Webcam frames are processed locally and are not recorded, saved, or uploaded.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime

import cv2

# ----------------------------- Configuration -----------------------------

CAMERA_INDEX = 0                 # Change to 1, 2, ... for another camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EMPTY_FRAMES_TO_RESET = 30       # Consecutive empty frames before re-arming

HOG_HIT_THRESHOLD = 0.0          # Higher = stricter; lower = more sensitive
HOG_SCALE = 1.05                 # Lower = more thorough but slower
HOG_WIN_STRIDE = (8, 8)
HOG_PADDING = (8, 8)
HOG_GROUP_THRESHOLD = 2

NOTIFICATION_TITLE = "Room Security Alert"
NOTIFICATION_MESSAGE = "A person has entered the monitored room!"
NOTIFICATION_TIMEOUT = 5

MAX_FRAME_FAILURES = 30
WINDOW_NAME = "Real-Time Room Monitoring System"

# -------------------------- Desktop notifications ------------------------

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    notification = None
    PLYER_AVAILABLE = False
    print("[WARNING] Plyer is unavailable; desktop notifications are disabled.")
    print("          Install dependencies with: python -m pip install -r requirements.txt")


def notification_worker(tasks: queue.Queue) -> None:
    """Show notifications on a background thread so video capture stays responsive."""
    while True:
        message = tasks.get()
        try:
            if message is None:
                return

            if PLYER_AVAILABLE:
                notification.notify(
                    title=NOTIFICATION_TITLE,
                    message=message,
                    app_name="Room Monitor",
                    timeout=NOTIFICATION_TIMEOUT,
                )
            else:
                print(f"\n[ROOM ALERT] {message}")
        except Exception as exc:
            print(f"[WARNING] Desktop notification failed: {exc}")
        finally:
            tasks.task_done()


# ------------------------------- Detection --------------------------------

def create_detector() -> cv2.HOGDescriptor:
    """Create OpenCV's built-in pretrained people detector."""
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def detect_people(frame, detector):
    """Return HOG detections as (x, y, width, height, score) tuples."""
    boxes, weights = detector.detectMultiScale(
        frame,
        hitThreshold=HOG_HIT_THRESHOLD,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE,
        groupThreshold=HOG_GROUP_THRESHOLD,
    )

    detections = []
    for box, weight in zip(boxes, weights):
        x, y, width, height = map(int, box)
        score = float(weight)
        detections.append((x, y, width, height, score))
    return detections


def draw_overlay(frame, detections, alert_active: bool, empty_count: int):
    """Draw bounding boxes, detector scores, and room status."""
    for x, y, width, height, score in detections:
        cv2.rectangle(
            frame, (x, y), (x + width, y + height), (0, 220, 0), 2
        )
        label = f"Person | HOG score: {score:.2f}"
        cv2.putText(
            frame, label, (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2,
        )

    person_detected = bool(detections)
    status = "PERSON DETECTED" if person_detected else "ROOM EMPTY"
    status_color = (0, 0, 255) if person_detected else (0, 220, 0)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (25, 25, 25), -1)
    cv2.putText(
        frame, status, (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2,
    )

    alert_label = "ALERT SENT" if alert_active else "MONITORING"
    cv2.putText(
        frame, alert_label, (frame.shape[1] - 165, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1,
    )

    if alert_active and not person_detected:
        cv2.putText(
            frame,
            f"Re-arming: {empty_count}/{EMPTY_FRAMES_TO_RESET} empty frames",
            (12, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1,
        )

    cv2.putText(
        frame, "Q / ESC: quit", (12, frame.shape[0] - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
    )
    return frame


# --------------------------------- Main ------------------------------------

def main() -> None:
    print("=" * 58)
    print("             REAL-TIME ROOM MONITOR")
    print("=" * 58)
    print("[INFO] Initializing HOG person detector...")

    try:
        detector = create_detector()
    except cv2.error as exc:
        print(f"[ERROR] Could not initialize the OpenCV detector: {exc}")
        return

    print(f"[INFO] Opening camera index {CAMERA_INDEX}...")
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("[ERROR] Webcam could not be opened.")
        print("        Close other camera apps and check CAMERA_INDEX.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    notification_tasks = queue.Queue()
    worker = threading.Thread(
        target=notification_worker,
        args=(notification_tasks,),
        name="notification-worker",
        daemon=True,
    )
    worker.start()

    alert_active = False
    empty_frame_count = 0
    failed_frame_count = 0

    print("[INFO] Monitoring started. Keep the room empty initially.")
    print("[INFO] Press Q or ESC in the video window to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                failed_frame_count += 1
                print(
                    f"[WARNING] Webcam frame read failed "
                    f"({failed_frame_count}/{MAX_FRAME_FAILURES})."
                )
                if failed_frame_count >= MAX_FRAME_FAILURES:
                    print("[ERROR] Too many consecutive camera failures; stopping.")
                    break
                time.sleep(0.05)
                continue

            failed_frame_count = 0

            # HOG works best when the frame is reasonably sized.
            detections = detect_people(frame, detector)
            person_detected = bool(detections)

            if person_detected:
                empty_frame_count = 0

                # Trigger once on the transition from empty to detected.
                if not alert_active:
                    alert_active = True
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[ALERT] Person detected at {timestamp}")
                    notification_tasks.put(NOTIFICATION_MESSAGE)

            elif alert_active:
                empty_frame_count += 1

                # Re-arm only after N consecutive frames with no detection.
                if empty_frame_count >= EMPTY_FRAMES_TO_RESET:
                    alert_active = False
                    empty_frame_count = 0
                    print("[INFO] Room empty; alert system re-armed.")

            frame = draw_overlay(
                frame, detections, alert_active, empty_frame_count
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Exit requested.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted from terminal.")
    except cv2.error as exc:
        print(f"[ERROR] OpenCV error: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

        # Ask the worker to finish, then wait briefly for a clean shutdown.
        notification_tasks.put(None)
        worker.join(timeout=2)

        print("[INFO] Camera released. Application closed.")


if __name__ == "__main__":
    main()
