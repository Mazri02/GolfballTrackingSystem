import cv2
import numpy as np
import time
import tempfile
import os
from picamera2 import Picamera2

# dp — Inverse ratio of accumulator resolution
# minDist — Minimum distance between circle centers
# param1 — Canny edge upper threshold
# param2 — Accumulator threshold (sensitivity)
# Min/Max Radius	Expected circle size
# rpuak_Xx67LEeDjdEvM4T9p6yu6RQK

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Initialize variables
last_detected_time = 0
detected_balls = []
snapshot_interval = 5  # seconds

# Temporary directory for snapshots
temp_dir = tempfile.TemporaryDirectory()
print(f"Temporary snapshot folder: {temp_dir.name}")

# Initial Hough circle parameters
params = {
    "dp": 12,          # scaled by /10
    "minDist": 40,
    "param1": 50,
    "param2": 30,
    "minRadius": 10,
    "maxRadius": 50
}

# Create window and sliders
cv2.namedWindow("Golfball Detection")

def nothing(x):
    pass

cv2.createTrackbar("dp x0.1", "Golfball Detection", params["dp"], 30, nothing)
cv2.createTrackbar("minDist", "Golfball Detection", params["minDist"], 200, nothing)
cv2.createTrackbar("param1", "Golfball Detection", params["param1"], 300, nothing)
cv2.createTrackbar("param2", "Golfball Detection", params["param2"], 150, nothing)
cv2.createTrackbar("Min Radius", "Golfball Detection", params["minRadius"], 100, nothing)
cv2.createTrackbar("Max Radius", "Golfball Detection", params["maxRadius"], 200, nothing)


def detect_golfballs(frame, dp, minDist, p1, p2, min_r, max_r):
    """Detect circular white-ish objects (golf balls)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=p1,
        param2=p2,
        minRadius=min_r,
        maxRadius=max_r
    )

    balls = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            balls.append((x, y, r))
    return balls


def draw_overlay(frame, balls):
    """Draw detected balls and their coordinates with readable background."""
    for (x, y, r) in balls:
        cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
        text = f"({x},{y})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        bg_x1, bg_y1 = x - 35, y - text_h - 15
        bg_x2, bg_y2 = bg_x1 + text_w + 10, bg_y1 + text_h + 10
        bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
        bg_x2, bg_y2 = min(frame.shape[1], bg_x2), min(frame.shape[0], bg_y2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, text, (bg_x1 + 5, bg_y2 - 5),
                    font, font_scale, (0, 255, 0), thickness)

    cv2.putText(frame, f"Balls: {len(balls)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def save_snapshot(frame, balls):
    """Save a temporary snapshot when new balls are detected."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(temp_dir.name, f"golfballs_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved: {filename} with {len(balls)} balls")


# Main loop
while True:
    frame = picam2.capture_array()

    # Read current slider values
    dp = max(0.1, cv2.getTrackbarPos("dp x0.1", "Golfball Detection") / 10.0)
    minDist = cv2.getTrackbarPos("minDist", "Golfball Detection")
    p1 = cv2.getTrackbarPos("param1", "Golfball Detection")
    p2 = cv2.getTrackbarPos("param2", "Golfball Detection")
    min_r = cv2.getTrackbarPos("Min Radius", "Golfball Detection")
    max_r = max(min_r + 1, cv2.getTrackbarPos("Max Radius", "Golfball Detection"))

    balls = detect_golfballs(frame, dp, minDist, p1, p2, min_r, max_r)

    # Detect if new balls appeared
    ball_positions = [(x, y) for (x, y, _) in balls]
    current_time = time.time()

    if set(ball_positions) != set(detected_balls):
        if current_time - last_detected_time > snapshot_interval:
            save_snapshot(frame, balls)
            last_detected_time = current_time
        detected_balls = ball_positions

    draw_overlay(frame, balls)
    cv2.imshow("Golfball Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
temp_dir.cleanup()
