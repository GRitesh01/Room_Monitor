# Real-Time Room Monitor

A beginner-friendly Python application that uses your local webcam and OpenCV's
pretrained HOG + SVM person detector. It draws a box around detected people and
sends one desktop notification when a person first appears. The alert resets
only after a configurable number of consecutive frames without a person.

**Privacy:** Video is processed locally. This application does not record,
save, or upload webcam footage.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or newer
- A working webcam

## Open and run in VS Code

1. Extract `Room_Monitor.zip`.
2. Open the extracted `Room_Monitor` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

5. Activate it.

   **Windows PowerShell:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Windows Command Prompt:**
   ```bat
   .venv\Scripts\activate.bat
   ```

   **macOS / Linux:**
   ```bash
   source .venv/bin/activate
   ```

6. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

7. Run:

   ```bash
   python main.py
   ```

8. Keep the room empty initially, then walk into the camera view to test.
9. Press **Q** or **Esc** while the video window is focused to quit.

## Configuration

Edit the constants near the top of `main.py`:

- `CAMERA_INDEX = 0`: default webcam. Try `1` or `2` for another camera.
- `EMPTY_FRAMES_TO_RESET = 30`: number of consecutive empty frames before
  another entry notification is allowed.
- `HOG_HIT_THRESHOLD = 0.0`: increase for stricter detection; decrease for
  more sensitive detection. Very low values may increase false positives.
- `HOG_SCALE = 1.05`: lower values scan more thoroughly but use more CPU.
- `NOTIFICATION_TITLE` and `NOTIFICATION_MESSAGE`: customize the popup text.

The displayed HOG score is a detector score, not a calibrated probability.

## Troubleshooting

### Webcam unavailable
- Close Zoom, Teams, browser camera tabs, or other apps using the camera.
- Check Windows **Settings → Privacy & security → Camera** and allow desktop
  apps to access the camera.
- Try changing `CAMERA_INDEX` from `0` to `1`.

### Missing dependencies
Activate the virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

### PowerShell will not activate the environment
Use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### Desktop notification does not appear
- Confirm notifications are enabled for your operating system.
- The application prints an alert in the terminal as a fallback.
- Some operating systems may suppress notifications while Focus Assist,
  Do Not Disturb, or a similar mode is enabled.

### Detection is slow or misses a person
HOG is a traditional full-body detector and is not as robust as modern
neural-network models. Use good lighting, keep the person reasonably visible,
and try a smaller `HOG_SCALE` such as `1.03` if your computer can handle it.

## Project files

```text
Room_Monitor/
├── main.py
├── requirements.txt
└── README.md
```

No model download is required: OpenCV provides the pretrained HOG people detector.
