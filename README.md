# Air Drawing with MediaPipe

Draw in the air using just your hand and a webcam. MediaPipe tracks 21 hand landmarks in real time to detect finger gestures â€” point to draw, peace sign to erase, fist to lift the pen.

## What it does

- Tracks your hand using MediaPipe's 21-point landmark model
- Detects three gestures: drawing (index finger up), erasing (peace sign), and pen lifted (fist)
- Draws smooth lines on a persistent canvas overlaid on the live webcam feed
- Lets you switch colors by hovering your finger over swatches at the top of the screen
- Applies position smoothing to reduce jitter from hand tremor

## Color palette

| Color | BGR Value |
|-------|-----------|
| Hot Pink | (180, 105, 255) |
| Lavender | (250, 180, 230) |
| Coral | (100, 130, 255) |
| Mint | (193, 240, 180) |
| Baby Blue | (255, 210, 180) |
| Lilac | (230, 160, 210) |

## Gestures

| Gesture | Action |
|---------|--------|
| â˜ï¸ Index finger up | Draw |
| âœŒï¸ Index + middle up | Erase |
| âœŠ All fingers down | Lift pen (stop drawing) |
| Hover over color swatch | Switch color |

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear canvas |
| `s` | Save drawing as `drawing.png` |

## Stack

- Python 3
- MediaPipe (hand landmark detection)
- OpenCV (webcam capture, drawing, display)
- NumPy (canvas array operations)

## Setup

```bash
pip install -r requirements.txt
python draw.py
```

## How it works

MediaPipe detects 21 landmarks on your hand each frame. The app checks whether each fingertip (landmark 8, 12, 16, 20) is above its middle joint (pip) to determine which fingers are extended. Index finger alone = draw mode. Index and middle together = erase mode. All fingers down = pen lifted.

A separate canvas array stores all drawn pixels. Each frame, pixels on the canvas are blended onto the webcam frame using a binary mask, so drawing persists across frames without affecting the live video underneath.

## Robotics connection

Hand gesture recognition is a key input modality for human-robot interaction. This project demonstrates real-time skeleton-based gesture classification using only an RGB camera â€” the same type of pipeline used in teleoperation interfaces and gesture-controlled robotic arms.


## Docker

`ash
docker build -t air-drawing-mediapipe .
docker run --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix air-drawing-mediapipe
`

> **Note:** Webcam and display passthrough requires Linux. On Windows/Mac, run natively instead.
