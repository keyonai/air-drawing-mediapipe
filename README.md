# Air Drawing with MediaPipe

Uses your webcam and hand gestures to draw in the air. Point to draw, peace sign to erase, fist to pick up the pen.

## Gestures

| Gesture | Action |
|---------|--------|
| Index finger up | Draw |
| Index + middle up | Erase |
| Fist | Lift pen |
| Hover over color swatch | Change color |

## Keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear canvas |
| `s` | Save as `drawing.png` |

## Run it

```bash
pip install -r requirements.txt
python draw.py
```

## Docker

```bash
docker build -t air-drawing-mediapipe .
docker run --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix air-drawing-mediapipe
```

> Webcam passthrough requires Linux. On Windows, just run it natively.
