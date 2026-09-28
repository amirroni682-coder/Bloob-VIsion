import cv2
import math
from collections import deque

# =========================================================
# COMPUTER VISION // MULTI OBJECT CONNECTED TRACKING
# =========================================================

print("=== MULTI OBJECT TRACKING ===")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MIN_BLOB_AREA = 700
MAX_BLOBS = 8

# Jarak maksimum antar-object agar dibuat garis
MAX_CONNECTION_DISTANCE = 220

# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Webcam:", cap.isOpened())

if not cap.isOpened():
    print("Kamera gagal dibuka")
    exit()

# ---------------------------------------------------------
# YUNET FACE DETECTOR
# ---------------------------------------------------------

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (CAMERA_WIDTH, CAMERA_HEIGHT),
    0.6,
    0.3,
    5000
)

# ---------------------------------------------------------
# MOTION DETECTOR
# ---------------------------------------------------------

background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

# ---------------------------------------------------------
# TRACK DATA
# ---------------------------------------------------------

face_history = deque(maxlen=20)

# ---------------------------------------------------------
# COLORS
# ---------------------------------------------------------

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
CYAN = (255, 220, 0)
BLUE = (255, 180, 0)
DARK = (20, 20, 20)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def draw_corner_box(
    frame,
    x,
    y,
    w,
    h,
    color,
    thickness=1,
    corner=12
):

    # TOP LEFT
    cv2.line(
        frame,
        (x, y),
        (x + corner, y),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x, y),
        (x, y + corner),
        color,
        thickness
    )

    # TOP RIGHT
    cv2.line(
        frame,
        (x + w, y),
        (x + w - corner, y),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x + w, y),
        (x + w, y + corner),
        color,
        thickness
    )

    # BOTTOM LEFT
    cv2.line(
        frame,
        (x, y + h),
        (x + corner, y + h),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x, y + h),
        (x, y + h - corner),
        color,
        thickness
    )

    # BOTTOM RIGHT
    cv2.line(
        frame,
        (x + w, y + h),
        (x + w - corner, y + h),
        color,
        thickness
    )

    cv2.line(
        frame,
        (x + w, y + h),
        (x + w, y + h - corner),
        color,
        thickness
    )


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        print("Gagal membaca kamera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # -----------------------------------------------------
    # FACE DETECTION
    # -----------------------------------------------------

    face_detector.setInputSize((w, h))

    _, faces = face_detector.detect(frame)

    face_center = None
    face_box = None

    if faces is not None and len(faces) > 0:

        # Ambil wajah terbesar
        faces = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True
        )

        face = faces[0]

        x = int(face[0])
        y = int(face[1])
        fw = int(face[2])
        fh = int(face[3])

        face_center = (
            x + fw // 2,
            y + fh // 2
        )

        face_box = (
            x,
            y,
            fw,
            fh
        )

        face_history.append(face_center)

        # FACE BOX
        draw_corner_box(
            frame,
            x,
            y,
            fw,
            fh,
            GREEN,
            1,
            14
        )

        # FACE LABEL
        cv2.putText(
            frame,
            "FACE TRACKING",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GREEN,
            1,
            cv2.LINE_AA
        )

        # FACE CENTER
        cv2.circle(
            frame,
            face_center,
            3,
            GREEN,
            -1
        )

    # -----------------------------------------------------
    # MOTION DETECTION
    # -----------------------------------------------------

    motion_mask = background.apply(frame)

    # Clean noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    motion_mask = cv2.dilate(
        motion_mask,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_BLOB_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        center = (
            x + bw // 2,
            y + bh // 2
        )

        blobs.append({
            "box": (x, y, bw, bh),
            "center": center,
            "area": area
        })

    # Ambil object terbesar
    blobs = sorted(
        blobs,
        key=lambda b: b["area"],
        reverse=True
    )

    blobs = blobs[:MAX_BLOBS]

    # -----------------------------------------------------
    # OBJECT NODES
    # -----------------------------------------------------

    object_points = []

    # Face menjadi NODE 0
    if face_center is not None:

        object_points.append({
            "point": face_center,
            "type": "face"
        })

    # Movement menjadi NODE berikutnya
    for blob in blobs:

        center = blob["center"]

        # Jangan deteksi seluruh wajah sebagai movement
        if face_box is not None:

            fx, fy, fw, fh = face_box

            if (
                fx < center[0] < fx + fw
                and
                fy < center[1] < fy + fh
            ):

                continue

        object_points.append({
            "point": center,
            "type": "object"
        })

    # -----------------------------------------------------
    # DRAW OBJECT SHAPES
    # -----------------------------------------------------

    for i, obj in enumerate(object_points):

        px, py = obj["point"]

        if obj["type"] == "face":

            # Face node
            cv2.circle(
                frame,
                (px, py),
                4,
                GREEN,
                -1
            )

        else:

            # Object shape
            size = 18

            draw_corner_box(
                frame,
                px - size,
                py - size,
                size * 2,
                size * 2,
                CYAN,
                1,
                6
            )

            cv2.circle(
                frame,
                (px, py),
                3,
                CYAN,
                -1
            )

    # -----------------------------------------------------
    # CONNECT OBJECTS
    # -----------------------------------------------------

    connections = []

    for i in range(len(object_points)):

        for j in range(i + 1, len(object_points)):

            p1 = object_points[i]["point"]
            p2 = object_points[j]["point"]

            d = distance(p1, p2)

            if d <= MAX_CONNECTION_DISTANCE:

                connections.append(
                    (i, j, d)
                )

    # Sort nearest connections first
    connections.sort(
        key=lambda c: c[2]
    )

    # Limit number of lines
    connections = connections[:12]

    for i, j, d in connections:

        p1 = object_points[i]["point"]
        p2 = object_points[j]["point"]

        # Line
        cv2.line(
            frame,
            p1,
            p2,
            CYAN,
            1,
            cv2.LINE_AA
        )

        # Small connection node
        midpoint = (
            (p1[0] + p2[0]) // 2,
            (p1[1] + p2[1]) // 2
        )

        cv2.circle(
            frame,
            midpoint,
            2,
            CYAN,
            -1
        )

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "FACE + BLOB TRACKING",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "TRACK ID : 01",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        WHITE,
        1,
        cv2.LINE_AA
    )

    # STATUS
    status = "FACE : DETECTED" if face_center else "FACE : SEARCHING"

    cv2.putText(
        frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        GREEN if face_center else WHITE,
        1,
        cv2.LINE_AA
    )

    # -----------------------------------------------------
    # MOVEMENT INFO
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"OBJECTS : {len(object_points)}",
        (20, h - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"CONNECTIONS : {len(connections)}",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        CYAN,
        1,
        cv2.LINE_AA
    )

    # -----------------------------------------------------
    # RIGHT SIDE INFO PANEL
    # -----------------------------------------------------

    if face_center is not None:

        px, py = face_center

        panel_x = min(
            w - 170,
            px + 120
        )

        panel_y = max(
            80,
            py - 50
        )

        panel_w = 145
        panel_h = 92

        # Panel
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (
                panel_x,
                panel_y
            ),
            (
                panel_x + panel_w,
                panel_y + panel_h
            ),
            DARK,
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.65,
            frame,
            0.35,
            0
        )

        # Connector from face to panel
        connector_start = (
            px + 8,
            py
        )

        connector_mid = (
            panel_x - 25,
            py
        )

        connector_end = (
            panel_x,
            panel_y + 30
        )

        cv2.line(
            frame,
            connector_start,
            connector_mid,
            CYAN,
            1,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            connector_mid,
            connector_end,
            CYAN,
            1,
            cv2.LINE_AA
        )

        # Panel text
        cv2.putText(
            frame,
            "TRACK ID : 01",
            (panel_x + 10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"X : {px}",
            (panel_x + 10, panel_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Y : {py}",
            (panel_x + 10, panel_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"LINK : {len(connections)}",
            (panel_x + 10, panel_y + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            GREEN,
            1,
            cv2.LINE_AA
        )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(
        "FACE + BLOB TRACKING",
        frame
    )

    # ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Tracking selesai.")
