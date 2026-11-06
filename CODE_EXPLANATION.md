# Room Monitor — Code Explanation, Portion by Portion

This document explains `main.py` from top to bottom. It is written for someone who is learning Python, OpenCV, and basic computer-vision application design.

## 1. What the application does

The application opens a webcam and continuously examines its video frames for people.

When a person is detected:

1. A bounding box is drawn around each detected person.
2. The screen status changes to **PERSON DETECTED**.
3. A desktop notification is queued the first time a person is detected in the current entry event.
4. Further detections do not create repeated notifications while the alert remains active.

When no person is detected, the application counts consecutive empty frames. Once the configured threshold is reached, it resets the alert so a later entry can trigger a new notification.

All video processing happens on the computer running the program. The code does not record, save, or upload webcam footage.

---

## 2. Imports — bringing in Python libraries

```python
import queue
import threading
import time
from datetime import datetime

import cv2
```

| Import | Purpose |
|---|---|
| `queue` | Safely passes notification messages from the video loop to the notification worker. |
| `threading` | Runs notifications in a separate background thread. |
| `time` | Adds a short pause after a failed camera read, avoiding a very tight retry loop. |
| `datetime` | Creates a readable timestamp when a person is detected. |
| `cv2` | OpenCV library: opens the webcam, detects people, draws graphics, and displays the video window. |

The program uses Python's built-in modules for queueing, threads, time, and timestamps. OpenCV is the main external computer-vision dependency.

---

## 3. Configuration — settings you can change

The configuration section keeps commonly changed values near the top of the file.

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

EMPTY_FRAMES_TO_RESET = 30

HOG_HIT_THRESHOLD = 0.0
HOG_SCALE = 1.05
HOG_WIN_STRIDE = (8, 8)
HOG_PADDING = (8, 8)
HOG_GROUP_THRESHOLD = 2

NOTIFICATION_TITLE = "Room Security Alert"
NOTIFICATION_MESSAGE = "A person has entered the monitored room!"
NOTIFICATION_TIMEOUT = 5

MAX_FRAME_FAILURES = 30
WINDOW_NAME = "Real-Time Room Monitoring System"
```

### Webcam settings

- `CAMERA_INDEX = 0`: selects the default webcam. If you have more than one camera, try `1` or `2`.
- `FRAME_WIDTH` and `FRAME_HEIGHT`: request a 640 × 480 capture size. The camera driver may choose a different supported size.

### Alert reset setting

`EMPTY_FRAMES_TO_RESET = 30` means the application must observe 30 consecutive frames with no detected person before it re-arms the alert.

The approximate reset time is:

```text
Reset time in seconds = required empty frames / actual processed frames per second
```

For example, at 30 processed frames per second, 30 empty frames take about one second. If detection runs at 10 frames per second, the same threshold takes about three seconds.

### HOG detector settings

- `HOG_HIT_THRESHOLD`: changes how strict the detector's acceptance threshold is. Increasing it generally makes acceptance stricter; lowering it may increase detections and false positives.
- `HOG_SCALE`: controls how much the detector image is resized between scanning scales. A value closer to `1.0` checks more scales and can be slower.
- `HOG_WIN_STRIDE`: controls the movement of the detector window. Smaller steps can be more thorough but slower.
- `HOG_PADDING`: adds a border around the image during detection.
- `HOG_GROUP_THRESHOLD`: controls how overlapping detections are grouped. Higher values can suppress weak or isolated detections.

These are tuning parameters, not a guaranteed sensitivity control. Test changes in the actual room and lighting conditions.

### Notification settings

- `NOTIFICATION_TITLE`: title shown in the desktop notification.
- `NOTIFICATION_MESSAGE`: message shown in the desktop notification.
- `NOTIFICATION_TIMEOUT`: requested display duration in seconds. The operating system may control the actual display duration.

---

## 4. Optional Plyer import — handling a missing notification package

```python
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    notification = None
    PLYER_AVAILABLE = False
```

`plyer` provides a convenient cross-platform desktop-notification interface.

The `try` / `except` block prevents the whole application from crashing if Plyer is missing:

- If the import works, `PLYER_AVAILABLE` becomes `True`.
- If it fails, desktop notifications are disabled and a warning is printed.
- The application can still run and print an alert in the terminal.

Install it with:

```bash
python -m pip install plyer
```

---

## 5. Notification worker — keeping popups out of the video loop

```python
def notification_worker(tasks: queue.Queue) -> None:
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
                print(f"\\n[ROOM ALERT] {message}")
        except Exception as exc:
            print(f"[WARNING] Desktop notification failed: {exc}")
        finally:
            tasks.task_done()
```

### Why use a queue and a thread?

Showing a desktop notification can take time or fail because of operating-system settings. If the webcam loop itself waited for the popup operation, the video could appear to freeze.

This design separates the jobs:

- **Main thread:** reads frames, detects people, draws the video, and checks keyboard input.
- **Notification thread:** waits for queued messages and displays notifications.

`tasks.get()` waits until a message is available. This is efficient: the worker does not need to continuously check an empty queue.

A `None` message is used as a shutdown signal. When the worker receives it, it exits.

`tasks.task_done()` marks each retrieved queue item as processed, including the shutdown item.

---

## 6. Creating the HOG person detector

```python
def create_detector() -> cv2.HOGDescriptor:
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )
    return detector
```

HOG means **Histogram of Oriented Gradients**. It describes patterns of edges and gradients in an image that can help distinguish human-shaped silhouettes.

OpenCV supplies a pretrained people detector:

```python
cv2.HOGDescriptor_getDefaultPeopleDetector()
```

The detector uses an SVM (Support Vector Machine) model that is already trained. You do not need to train or download a separate model for this basic version.

The function returns the configured detector so it can be reused for every frame.

**Important:** HOG is a traditional full-body person detector. It may miss people who are partly hidden, very close to the camera, poorly lit, or outside its expected viewing conditions. It can also produce false positives.

---

## 7. Detecting people in a frame

```python
def detect_people(frame, detector):
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
```

### What is `detectMultiScale()`?

It scans the image at multiple sizes to find people who may appear at different scales in the camera view.

It returns:

- `boxes`: the detected rectangles.
- `weights`: detector scores associated with those rectangles.

Each rectangle contains:

- `x`: left coordinate.
- `y`: top coordinate.
- `width`: rectangle width.
- `height`: rectangle height.

The code converts the coordinates to integers and the score to a Python float. It then stores each detection as:

```text
(x, y, width, height, score)
```

The returned list is empty when the detector finds no people.

### Understanding the HOG score

The displayed score is the detector's raw HOG/SVM score. It is **not** a calibrated probability. A score of `0.90` does not mean there is a 90% chance that the rectangle contains a person.

---

## 8. Drawing boxes and status information

The `draw_overlay()` function receives the current video frame and the detections.

### Bounding boxes

```python
cv2.rectangle(
    frame, (x, y), (x + width, y + height), (0, 220, 0), 2
)
```

OpenCV draws a rectangle from its top-left point to its bottom-right point.

The color is supplied in **BGR** order, not RGB:

- `(0, 220, 0)` is green.
- `(0, 0, 255)` is red.
- `(255, 255, 255)` is white.

The final argument `2` is the line thickness.

### Detection label

```python
label = f"Person | HOG score: {score:.2f}"
```

The `:.2f` formatting displays the score with two digits after the decimal point.

`cv2.putText()` draws the label on the frame.

### Room status

```python
person_detected = bool(detections)
status = "PERSON DETECTED" if person_detected else "ROOM EMPTY"
```

A non-empty detection list evaluates to `True`; an empty list evaluates to `False`.

The function displays:

- **PERSON DETECTED** in red when at least one person is detected.
- **ROOM EMPTY** in green when no person is detected.

It also displays whether an alert has already been sent. While the alert is active but the person is temporarily not detected, it displays the empty-frame reset progress.

Finally, the function returns the modified frame for display.

---

## 9. Main function — starting the application

```python
def main() -> None:
    detector = create_detector()
    camera = cv2.VideoCapture(CAMERA_INDEX)
```

The `main()` function coordinates the application.

First, it creates the HOG detector. Then it asks OpenCV to open the selected webcam.

### Checking whether the webcam opened

```python
if not camera.isOpened():
    print("[ERROR] Webcam could not be opened.")
    return
```

This check handles common problems such as:

- No camera connected.
- Wrong camera index.
- Another application currently using the camera.
- Operating-system camera permissions being denied.

If the camera cannot be opened, the program prints an error and exits instead of entering the monitoring loop.

### Requesting the capture size

```python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
```

These lines request the configured capture dimensions. Some webcams or drivers may ignore the request or select a nearby supported resolution.

---

## 10. Starting the notification thread

```python
notification_tasks = queue.Queue()

worker = threading.Thread(
    target=notification_worker,
    args=(notification_tasks,),
    name="notification-worker",
    daemon=True,
)
worker.start()
```

A queue is created to hold notification messages.

A thread is then created with:

- `target`: the function the thread should run.
- `args`: the queue passed to that function.
- `name`: a helpful thread name for debugging.
- `daemon=True`: the worker is a background thread and will not keep the Python process alive by itself after the main program exits.

`worker.start()` starts the worker. The main thread can now continue with webcam processing.

---

## 11. Detection state variables

```python
alert_active = False
empty_frame_count = 0
failed_frame_count = 0
```

These variables control the alert behavior.

| Variable | Meaning |
|---|---|
| `alert_active` | `False` means a new detection can trigger an alert. `True` means the current entry alert has already been sent. |
| `empty_frame_count` | Counts consecutive frames in which no person is detected while an alert is active. |
| `failed_frame_count` | Counts consecutive webcam frame-read failures. |

At startup, the alert is inactive, so the first person detection can trigger a notification.

---

## 12. The webcam monitoring loop

```python
while True:
    ok, frame = camera.read()
```

`camera.read()` asks OpenCV for the next frame.

It returns two values:

- `ok`: whether the read succeeded.
- `frame`: the image data.

If a read fails, the application increments `failed_frame_count`, prints a warning, and briefly sleeps before trying again.

If the number of consecutive failures reaches `MAX_FRAME_FAILURES`, the loop stops. A successful read resets the failure counter to zero.

This prevents the program from running forever when the camera stops supplying frames.

---

## 13. Running detection on every frame

```python
detections = detect_people(frame, detector)
person_detected = bool(detections)
```

The current frame is passed to the HOG detector.

The `detections` list contains all detected people. Converting it to `bool` gives a simple state:

- `True`: at least one person was detected.
- `False`: no person was detected.

The application does not identify who the person is. It only attempts to detect human figures.

---

## 14. Entry event — send only one notification

```python
if person_detected:
    empty_frame_count = 0

    if not alert_active:
        alert_active = True
        notification_tasks.put(NOTIFICATION_MESSAGE)
```

This is the key anti-repeat logic.

When a person is detected:

1. The empty-frame counter is reset to zero.
2. The program checks whether an alert is already active.
3. If no alert is active, it activates the alert and queues one notification.
4. If an alert is already active, it does not queue another notification.

The timestamp is also printed to the terminal for a simple event record. It is not written to a file or database.

### Example

Suppose a person remains visible for 500 frames:

- Frame 1: person detected; one notification is queued.
- Frames 2–500: person remains detected; no further notification is queued.

This prevents notification spam while the person remains in view.

---

## 15. Empty-room frames — re-arm the alert

```python
elif alert_active:
    empty_frame_count += 1

    if empty_frame_count >= EMPTY_FRAMES_TO_RESET:
        alert_active = False
        empty_frame_count = 0
```

When no person is detected and the alert is active, the empty-frame counter increases.

Once the counter reaches `EMPTY_FRAMES_TO_RESET`, the application:

1. Sets `alert_active` back to `False`.
2. Resets the empty-frame counter to zero.
3. Prints that the room is empty and the alert has been re-armed.

If a person is detected again before the threshold is reached, the earlier person-detected branch resets `empty_frame_count` to zero. Therefore, the empty frames must be consecutive.

This is useful because a detector can occasionally miss a person for one or two frames due to movement, lighting, or partial obstruction.

---

## 16. Displaying the video and reading keyboard input

```python
frame = draw_overlay(
    frame, detections, alert_active, empty_frame_count
)
cv2.imshow(WINDOW_NAME, frame)

key = cv2.waitKey(1) & 0xFF
if key in (ord("q"), 27):
    break
```

- `draw_overlay()` adds boxes, labels, and status information.
- `cv2.imshow()` displays the frame in a desktop window.
- `cv2.waitKey(1)` waits briefly for a keyboard event and allows the OpenCV window to refresh.
- `ord("q")` represents the Q key.
- `27` is the Escape key code.
- `break` exits the monitoring loop.

The OpenCV window must be focused for its keyboard shortcuts to work reliably.

---

## 17. Error handling and safe cleanup

The main loop is wrapped in `try` / `except` / `finally`.

### Keyboard interruption

```python
except KeyboardInterrupt:
    print("\\n[INFO] Interrupted from terminal.")
```

This handles Ctrl+C in the terminal.

### OpenCV errors and unexpected errors

The application prints a useful message rather than silently failing.

### The `finally` block

```python
finally:
    camera.release()
    cv2.destroyAllWindows()

    notification_tasks.put(None)
    worker.join(timeout=2)
```

Cleanup runs whether the loop ends normally or an exception occurs:

1. `camera.release()` releases the webcam.
2. `cv2.destroyAllWindows()` closes OpenCV windows.
3. `notification_tasks.put(None)` tells the notification worker to stop.
4. `worker.join(timeout=2)` waits up to two seconds for the worker to finish.

Releasing the webcam is important because other applications may need to use it later.

---

## 18. Python entry point

```python
if __name__ == "__main__":
    main()
```

Python sets `__name__` to `"__main__"` when a file is run directly.

This condition means `main()` starts when you execute:

```bash
python main.py
```

It also means the file can be imported by another Python file without automatically starting the webcam application.

---

## 19. Complete detection-state example

The following table illustrates how the alert state changes.

| Event | Person detected? | Empty-frame count | Alert active? | Notification |
|---|---:|---:|---:|---|
| Application starts | No | 0 | No | No |
| Person enters | Yes | 0 | Yes | **Send** |
| Person remains | Yes | 0 | Yes | No |
| Brief missed detection | No | 1 | Yes | No |
| Person detected again | Yes | 0 | Yes | No |
| Person leaves | No | 1, 2, 3… | Yes | No |
| 30 consecutive empty frames | No | Reset to 0 | No | No |
| Person enters again | Yes | 0 | Yes | **Send again** |

The reset count is configurable. The example uses 30 frames.

---

## 20. Installation and running commands

Open the project folder in VS Code and use its terminal.

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Activate on Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the application

```bash
python main.py
```

### Check OpenCV installation

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

## 21. Known limitations

This is a useful learning prototype, not a complete security product.

1. **HOG is not a modern high-accuracy detector.** It can miss people or detect person-like shapes incorrectly.
2. **A camera-view detection is not proof of doorway entry.** The current program detects a person appearing anywhere in the camera's field of view.
3. **No identity recognition is performed.** The program does not identify or recognize individuals.
4. **No persistent event history exists.** Timestamps are printed in the terminal only.
5. **No recording exists.** Frames are processed in memory and are not saved by this code.
6. **HOG score is not a probability.** Do not interpret it as a percentage confidence.
7. **Notification delivery depends on the operating system.** Do Not Disturb settings or notification permissions can suppress desktop popups.

For a later version, consider a YOLO-based person detector, object tracking, and a defined doorway region. Those improvements can help distinguish a person entering from someone already present or merely passing through the camera view.

---

## 22. Suggested learning order

If you are new to this project, study and test it in this order:

1. Understand the imports and configuration.
2. Run the program and confirm the webcam opens.
3. Understand `camera.read()` and the video loop.
4. Study `create_detector()` and `detect_people()`.
5. Understand how bounding boxes are drawn.
6. Trace `alert_active` and `empty_frame_count` manually.
7. Temporarily change `EMPTY_FRAMES_TO_RESET` to a small value for testing.
8. Study the notification queue and background thread.
9. Test camera failure handling and safe exit.
10. Only then begin adding more advanced detection or tracking.

**Recommended first experiment:** Set `EMPTY_FRAMES_TO_RESET = 60`, run the program, enter the camera view, leave the view for long enough to reset, and enter again. You should receive one notification for each separate entry event.
