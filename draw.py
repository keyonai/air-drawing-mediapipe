import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────
# GIRLY COLOR PALETTE (BGR format for OpenCV)
# ─────────────────────────────────────────────
COLORS = {
    "Hot Pink":    (180, 105, 255),   # hot pink
    "Lavender":    (250, 180, 230),   # soft lavender
    "Coral":       (100, 130, 255),   # coral/salmon
    "Mint":        (193, 240, 180),   # mint green
    "Baby Blue":   (255, 210, 180),   # baby blue
    "Lilac":       (230, 160, 210),   # lilac purple
}
COLOR_NAMES = list(COLORS.keys())
COLOR_VALUES = list(COLORS.values())

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
BRUSH_SIZE      = 8      # thickness of drawn lines
ERASER_SIZE     = 40     # thickness of eraser
TIP_SMOOTHING   = 0.5    # smoothing factor for fingertip position (0=none, 1=max)

# ─────────────────────────────────────────────
# MEDIAPIPE HAND SETUP
# ─────────────────────────────────────────────
# MediaPipe Hands detects 21 landmarks on each hand
# Landmark indices we use:
#   4  = thumb tip
#   8  = index finger tip
#   12 = middle finger tip
#   16 = ring finger tip
#   20 = pinky tip
#   6  = index finger pip (middle joint) — used to check if finger is up
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,          # track one hand only
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)


def fingers_up(landmarks, w, h):
    """
    Returns a list of 5 booleans [thumb, index, middle, ring, pinky]
    True = finger is extended (pointing up), False = finger is curled.
    We check if each fingertip is above its middle joint (pip).
    """
    tips = [4, 8, 12, 16, 20]   # tip landmark indices
    pips = [3, 6, 10, 14, 18]   # pip (middle joint) landmark indices

    up = []
    for tip, pip in zip(tips, pips):
        # y coordinate: smaller = higher on screen
        tip_y = landmarks[tip].y * h
        pip_y = landmarks[pip].y * h
        up.append(tip_y < pip_y)   # tip is above pip = finger is up

    return up  # [thumb, index, middle, ring, pinky]


def draw_color_palette(frame, current_color_idx):
    """
    Draws color swatches along the top of the frame.
    Hover your index finger over a swatch to switch colors.
    """
    swatch_w = 80
    swatch_h = 50
    margin   = 10

    for i, (name, color) in enumerate(COLORS.items()):
        x1 = margin + i * (swatch_w + margin)
        y1 = margin
        x2 = x1 + swatch_w
        y2 = y1 + swatch_h

        # Draw filled swatch
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Highlight currently selected color with white border
        if i == current_color_idx:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 3)

        # Draw color name
        cv2.putText(frame, name, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return frame


def check_color_selection(finger_x, finger_y, current_color_idx):
    """
    If the fingertip is hovering over a color swatch, switch to that color.
    """
    swatch_w = 80
    swatch_h = 50
    margin   = 10

    for i in range(len(COLORS)):
        x1 = margin + i * (swatch_w + margin)
        y1 = margin
        x2 = x1 + swatch_w
        y2 = y1 + swatch_h

        if x1 < finger_x < x2 and y1 < finger_y < y2:
            return i   # switch to this color

    return current_color_idx   # no change


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # ── Canvas: transparent overlay where drawing is stored ──
    # We keep a separate canvas and blend it onto the webcam frame each tick
    ret, frame = cap.read()
    canvas = np.zeros_like(frame)   # black canvas, same size as webcam

    current_color_idx = 0    # start with Hot Pink
    prev_x, prev_y    = 0, 0 # previous fingertip position for line drawing
    smoothed_x        = 0    # smoothed fingertip x position
    smoothed_y        = 0    # smoothed fingertip y position

    print("Controls:")
    print("  ✦ Point (index finger only) → draw")
    print("  ✦ Fist (all fingers down)   → stop drawing / lift pen")
    print("  ✦ Peace sign (index + middle up) → eraser mode")
    print("  ✦ Hover over color swatches at top to switch color")
    print("  ✦ Press 'c' to clear canvas")
    print("  ✦ Press 'q' to quit")
    print("  ✦ Press 's' to save drawing")

    eraser_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Run MediaPipe hand detection ──
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "none"   # "drawing", "erasing", "idle", or "none"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                # Draw the hand skeleton (landmarks + connections)
                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                up = fingers_up(landmarks, w, h)
                # up = [thumb, index, middle, ring, pinky]

                # Get index fingertip position in pixel coordinates
                raw_x = int(landmarks[8].x * w)
                raw_y = int(landmarks[8].y * h)

                # Smooth the fingertip position to reduce jitter
                smoothed_x = int(smoothed_x * TIP_SMOOTHING + raw_x * (1 - TIP_SMOOTHING))
                smoothed_y = int(smoothed_y * TIP_SMOOTHING + raw_y * (1 - TIP_SMOOTHING))

                # ── Detect gesture ──
                index_up  = up[1]
                middle_up = up[2]
                ring_up   = up[3]
                pinky_up  = up[4]

                all_down = not index_up and not middle_up and not ring_up and not pinky_up

                if index_up and middle_up and not ring_up and not pinky_up:
                    # ✌️ Peace sign = ERASER
                    gesture = "erasing"
                    eraser_mode = True
                elif index_up and not middle_up and not ring_up and not pinky_up:
                    # ☝️ One finger = DRAW
                    gesture = "drawing"
                    eraser_mode = False

                    # Check if hovering over color palette
                    if smoothed_y < 70:
                        current_color_idx = check_color_selection(
                            smoothed_x, smoothed_y, current_color_idx)
                        prev_x, prev_y = 0, 0  # reset so we don't draw a line
                    else:
                        # Draw line from previous point to current point
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = smoothed_x, smoothed_y

                        cv2.line(canvas,
                                 (prev_x, prev_y),
                                 (smoothed_x, smoothed_y),
                                 COLOR_VALUES[current_color_idx],
                                 BRUSH_SIZE)
                        prev_x, prev_y = smoothed_x, smoothed_y

                elif all_down:
                    # ✊ Fist = LIFT PEN (stop drawing)
                    gesture = "idle"
                    prev_x, prev_y = 0, 0

                else:
                    prev_x, prev_y = 0, 0

                # ── Erase mode ──
                if eraser_mode and gesture == "erasing":
                    cv2.circle(canvas, (smoothed_x, smoothed_y),
                               ERASER_SIZE, (0, 0, 0), -1)
                    prev_x, prev_y = 0, 0

                # ── Draw fingertip dot ──
                dot_color = (255, 255, 255) if eraser_mode else COLOR_VALUES[current_color_idx]
                cv2.circle(frame, (smoothed_x, smoothed_y), 8, dot_color, -1)

        else:
            # No hand detected — reset prev position
            prev_x, prev_y = 0, 0

        # ── Blend canvas onto webcam frame ──
        # Where canvas is not black (drawn pixels), overlay them on the frame
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
        frame[mask > 0] = canvas[mask > 0]

        # ── Draw color palette at top ──
        frame = draw_color_palette(frame, current_color_idx)

        # ── Draw gesture status ──
        status_map = {
            "drawing": f"Drawing  ({COLOR_NAMES[current_color_idx]})",
            "erasing": "Erasing",
            "idle":    "Pen Lifted",
            "none":    "No Hand",
        }
        status_color_map = {
            "drawing": COLOR_VALUES[current_color_idx],
            "erasing": (255, 255, 255),
            "idle":    (180, 180, 180),
            "none":    (100, 100, 100),
        }
        cv2.putText(frame, status_map[gesture],
                    (w - 260, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    status_color_map[gesture], 2)

        cv2.imshow("Air Drawing  |  q=quit  c=clear  s=save", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the canvas
            canvas = np.zeros_like(frame)
            print("Canvas cleared!")
        elif key == ord('s'):
            # Save the drawing
            cv2.imwrite("drawing.png", canvas)
            print("Saved drawing to drawing.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
