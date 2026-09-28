import cv2
import numpy as np
import time
import math
from collections import deque, Counter

from ultralytics import YOLO
import mediapipe as mp


# ============================================================
# CONFIG
# ============================================================

FACE_MODEL = "models/face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = "models/emotion-ferplus-8.onnx"
YOLO_MODEL = "yolo11n.pt"


# ----------------------------
# FACE
# ----------------------------

MAX_FACE_DISTANCE = 150
MAX_FACE_MISSING_FRAMES = 5

MOVEMENT_THRESHOLD = 5.0
STABLE_SPEED_THRESHOLD = 12.0

POSITION_SMOOTHING = 0.35
SPEED_SMOOTHING = 0.25

DIRECTION_CONFIRM_FRAMES = 3

EXPRESSION_HISTORY_LENGTH = 8


# ----------------------------
# OBJECT
# ----------------------------

OBJECT_CONFIDENCE = 0.35
MAX_OBJECT_DISTANCE = 120
MAX_OBJECT_MISSING_FRAMES = 10


# ----------------------------
# HAND
# ----------------------------

MAX_HAND_DISTANCE = 180
MAX_HAND_MISSING_FRAMES = 8


# ----------------------------
# YOLO
# ----------------------------

YOLO_INTERVAL = 2


# ============================================================
# COLORS
# ============================================================

CYAN = (255, 220, 80)
BLUE = (255, 130, 40)
LIME = (120, 255, 120)
AMBER = (80, 190, 255)
MAGENTA = (255, 100, 220)

WHITE = (255, 255, 255)
GRAY = (150, 150, 150)

DARK = (18, 22, 28)


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTIONS = [
    "NEUTRAL",
    "HAPPY",
    "SURPRISE",
    "SAD",
    "ANGRY",
    "DISGUST",
    "FEAR",
    "CONTEMPT"
]


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# TRACKING ENGINE
# ============================================================

class TrackingEngine:

    def __init__(self):

        print("====================================")
        print("INITIALIZING BLOB TRACKING ENGINE")
        print("====================================")

        # ----------------------------------------------------
        # TRACKS
        # ----------------------------------------------------

        self.face_tracks = {}
        self.next_face_id = 1

        self.object_tracks = {}
        self.next_object_id = 1

        self.hand_tracks = {}
        self.next_hand_id = 1

        self.frame_counter = 0

        self.previous_time = time.time()
        self.fps = 0.0
        self.effect = "medium"

        # Runtime visual style. Colors are OpenCV BGR.
        self.face_color = CYAN
        self.hand_color = LIME
        self.object_color = AMBER
        self.text_color = (8, 12, 16)
        self.text_scale = 1.0
        self.box_thickness = 2
        self.line_thickness = 2

        # ----------------------------------------------------
        # LOAD YUNET
        # ----------------------------------------------------

        print("Loading YuNet face detector...")

        self.face_detector = cv2.FaceDetectorYN.create(
            FACE_MODEL,
            "",
            (320, 320),
            0.65,
            0.3,
            5000
        )

        # ----------------------------------------------------
        # LOAD FER+
        # ----------------------------------------------------

        print("Loading FER+ emotion model...")

        self.emotion_net = cv2.dnn.readNetFromONNX(
            EMOTION_MODEL
        )

        # ----------------------------------------------------
        # LOAD YOLO
        # ----------------------------------------------------

        print("Loading YOLO...")

        self.yolo_model = YOLO(
            YOLO_MODEL
        )

        # ----------------------------------------------------
        # LOAD MEDIAPIPE
        # ----------------------------------------------------

        print("Loading MediaPipe Hands...")

        mp_hands = mp.solutions.hands

        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55
        )

        print("Tracking engine ready.")


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.face_tracks = {}
        self.next_face_id = 1

        self.object_tracks = {}
        self.next_object_id = 1

        self.hand_tracks = {}
        self.next_hand_id = 1

        self.frame_counter = 0

        self.previous_time = time.time()

        self.fps = 0.0
        self.effect = "medium"


    def set_effect(self, effect):

        effect = str(effect).lower()

        if effect not in ("low", "medium", "high"):
            effect = "medium"

        self.effect = effect


    @staticmethod
    def hex_to_bgr(value, fallback):
        """Convert #RRGGBB / RRGGBB into OpenCV BGR."""
        try:
            text = str(value).strip().lstrip("#")
            if len(text) != 6:
                return fallback
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
            return (b, g, r)
        except Exception:
            return fallback


    def set_style(
        self,
        face_color=None,
        hand_color=None,
        object_color=None,
        text_color=None,
        text_size=None,
        box_size=None,
        line_size=None,
    ):
        """Apply runtime overlay styling without changing tracking logic."""
        if face_color is not None:
            self.face_color = self.hex_to_bgr(face_color, self.face_color)

        if hand_color is not None:
            self.hand_color = self.hex_to_bgr(hand_color, self.hand_color)

        if object_color is not None:
            self.object_color = self.hex_to_bgr(object_color, self.object_color)

        if text_color is not None:
            self.text_color = self.hex_to_bgr(text_color, self.text_color)

        if text_size is not None:
            try:
                value = float(text_size)
                self.text_scale = max(0.35, min(value / 0.45, 2.4))
            except (TypeError, ValueError):
                pass

        if box_size is not None:
            try:
                self.box_thickness = max(1, min(int(round(float(box_size))), 8))
            except (TypeError, ValueError):
                pass

        if line_size is not None:
            try:
                self.line_thickness = max(1, min(int(round(float(line_size))), 8))
            except (TypeError, ValueError):
                pass


    def get_stats(self):

        visible_faces = sum(
            1 for t in self.face_tracks.values()
            if t.get("missing_frames", 0) == 0
        )

        visible_hands = sum(
            1 for t in self.hand_tracks.values()
            if t.get("missing_frames", 0) == 0
        )

        visible_objects = sum(
            1 for t in self.object_tracks.values()
            if t.get("missing_frames", 0) == 0
        )

        return {
            "fps": round(float(self.fps), 1),
            "face": visible_faces,
            "hand": visible_hands,
            "object": visible_objects,
            "effect": self.effect,
        }


    # ========================================================
    # HELPER
    # ========================================================

    @staticmethod
    def distance(p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )


    @staticmethod
    def clamp(
        value,
        minimum,
        maximum
    ):

        return max(
            minimum,
            min(value, maximum)
        )


    @staticmethod
    def draw_text(
        frame,
        text,
        position,
        color=WHITE,
        scale=0.45,
        thickness=1
    ):

        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )


    @staticmethod
    def draw_panel(
        frame,
        x,
        y,
        width,
        height,
        border_color=CYAN,
        alpha=0.82
    ):

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            DARK,
            -1
        )

        frame[:] = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            border_color,
            1
        )


    @staticmethod
    def draw_plus(
        frame,
        center,
        color=CYAN,
        size=8,
        thickness=2
    ):

        cx = int(center[0])
        cy = int(center[1])

        cv2.line(
            frame,
            (cx - size, cy),
            (cx + size, cy),
            color,
            thickness,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (cx, cy - size),
            (cx, cy + size),
            color,
            thickness,
            cv2.LINE_AA
        )


    @staticmethod
    def draw_reference_box(frame, x1, y1, x2, y2, color, thickness=2, corners=False):

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)

        if corners:
            corner = max(8, int(min(abs(x2 - x1), abs(y2 - y1)) * 0.08))
            corner = min(corner, 24)
            cv2.line(frame, (int(x1), int(y1)), (int(x1 + corner), int(y1)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1 + corner)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x2), int(y1)), (int(x2 - corner), int(y1)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x2), int(y1)), (int(x2), int(y1 + corner)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x1), int(y2)), (int(x1 + corner), int(y2)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x1), int(y2)), (int(x1), int(y2 - corner)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x2), int(y2)), (int(x2 - corner), int(y2)), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (int(x2), int(y2)), (int(x2), int(y2 - corner)), color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_reference_label(frame, text, x, y, color, scale=0.42, text_color=None, padding_x=6, padding_y=4):

        if text_color is None:
            text_color = (8, 12, 16)

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = int(max(0, x))
        y = int(max(th + padding_y * 2, y))
        left = x
        top = max(0, y - th - padding_y * 2)
        right = min(frame.shape[1] - 1, x + tw + padding_x * 2)
        bottom = y

        overlay = frame.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
        frame[:] = cv2.addWeighted(overlay, 0.90, frame, 0.10, 0)

        cv2.putText(frame, text, (left + padding_x, bottom - padding_y - baseline), font, scale, text_color, thickness, cv2.LINE_AA)


    # ========================================================
    # EMOTION
    # ========================================================

    def detect_emotion(
        self,
        face_crop
    ):

        if face_crop is None:
            return "UNKNOWN", 0.0

        if face_crop.size == 0:
            return "UNKNOWN", 0.0

        try:

            gray = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                (64, 64)
            )

            gray = gray.astype(
                np.float32
            ) / 255.0

            blob = cv2.dnn.blobFromImage(
                gray,
                scalefactor=1.0,
                size=(64, 64),
                mean=(0,),
                swapRB=False,
                crop=False
            )

            self.emotion_net.setInput(
                blob
            )

            prediction = (
                self.emotion_net.forward()
            )

            prediction = prediction.reshape(
                -1
            )

            exp_prediction = np.exp(
                prediction -
                np.max(prediction)
            )

            probabilities = (
                exp_prediction /
                np.sum(exp_prediction)
            )

            index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[index]
            )

            return (
                EMOTIONS[index],
                confidence
            )

        except Exception as e:

            print(
                "Emotion error:",
                e
            )

            return (
                "UNKNOWN",
                0.0
            )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        self.face_detector.setInputSize(
            (width, height)
        )

        _, faces = self.face_detector.detect(
            frame
        )

        results = []

        if faces is None:
            return results

        for face in faces:

            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])

            x = self.clamp(
                x,
                0,
                width - 1
            )

            y = self.clamp(
                y,
                0,
                height - 1
            )

            w = self.clamp(
                w,
                1,
                width - x
            )

            h = self.clamp(
                h,
                1,
                height - y
            )

            center = (
                x + w / 2,
                y + h / 2
            )

            detection_confidence = 0.0
            try:
                if len(face) >= 15:
                    detection_confidence = float(face[14])
            except Exception:
                detection_confidence = 0.0

            results.append({

                "bbox": (
                    x,
                    y,
                    w,
                    h
                ),

                "center": center,

                "confidence": detection_confidence
            })


        # ----------------------------------------------------
        # REMOVE DUPLICATE FACE
        # ----------------------------------------------------

        filtered = []

        for candidate in results:

            duplicate = False

            for existing in filtered:

                if self.distance(
                    candidate["center"],
                    existing["center"]
                ) < 45:

                    duplicate = True
                    break

            if not duplicate:

                filtered.append(
                    candidate
                )

        return filtered


    # ========================================================
    # UPDATE FACE TRACKS
    # ========================================================

    def update_face_tracks(
        self,
        detections,
        frame
    ):

        current_time = time.time()

        matched_ids = set()

        for detection in detections:

            center = detection["center"]

            best_id = None

            best_distance = (
                MAX_FACE_DISTANCE
            )

            for face_id, track in (
                self.face_tracks.items()
            ):

                if face_id in matched_ids:
                    continue

                d = self.distance(
                    center,
                    track["center"]
                )

                if d < best_distance:

                    best_distance = d
                    best_id = face_id


            # ------------------------------------------------
            # NEW FACE
            # ------------------------------------------------

            if best_id is None:

                face_id = self.next_face_id

                self.next_face_id += 1

                x, y, w, h = (
                    detection["bbox"]
                )

                face_crop = frame[
                    y:y + h,
                    x:x + w
                ]

                emotion, confidence = (
                    self.detect_emotion(
                        face_crop
                    )
                )

                self.face_tracks[
                    face_id
                ] = {

                    "id": face_id,

                    "bbox": detection["bbox"],

                    "confidence": detection.get("confidence", 0.0),

                    "center": center,

                    "raw_center": center,

                    "previous_center": center,

                    "velocity": (
                        0.0,
                        0.0
                    ),

                    "speed": 0.0,

                    "avg_speed": 0.0,

                    "max_speed": 0.0,

                    "total_distance": 0.0,

                    "direction": "STABLE",

                    "candidate_direction": "STABLE",

                    "direction_count": 0,

                    "status": "STABLE",

                    "emotion": emotion,

                    "emotion_confidence": confidence,

                    "emotion_history": deque(
                        [emotion],
                        maxlen=EXPRESSION_HISTORY_LENGTH
                    ),

                    "confidence_history": deque(
                        [confidence],
                        maxlen=EXPRESSION_HISTORY_LENGTH
                    ),

                    "history": deque(
                        [center],
                        maxlen=60
                    ),

                    "created_at": current_time,

                    "last_seen": current_time,

                    "missing_frames": 0
                }

                matched_ids.add(
                    face_id
                )

                continue


            # ------------------------------------------------
            # EXISTING FACE
            # ------------------------------------------------

            track = self.face_tracks[
                best_id
            ]

            old_center = track[
                "center"
            ]

            smooth_x = (
                old_center[0]
                *
                (1 - POSITION_SMOOTHING)
                +
                center[0]
                *
                POSITION_SMOOTHING
            )

            smooth_y = (
                old_center[1]
                *
                (1 - POSITION_SMOOTHING)
                +
                center[1]
                *
                POSITION_SMOOTHING
            )

            new_center = (
                smooth_x,
                smooth_y
            )

            dx = (
                new_center[0]
                -
                old_center[0]
            )

            dy = (
                new_center[1]
                -
                old_center[1]
            )

            movement = math.sqrt(
                dx * dx +
                dy * dy
            )

            speed = (
                track["speed"]
                *
                (1 - SPEED_SMOOTHING)
                +
                movement
                *
                SPEED_SMOOTHING
            )


            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            if movement < MOVEMENT_THRESHOLD:

                detected_direction = "STABLE"

            elif abs(dx) > abs(dy):

                detected_direction = (
                    "RIGHT"
                    if dx > 0
                    else
                    "LEFT"
                )

            else:

                detected_direction = (
                    "DOWN"
                    if dy > 0
                    else
                    "UP"
                )


            if (
                detected_direction
                ==
                track["candidate_direction"]
            ):

                track[
                    "direction_count"
                ] += 1

            else:

                track[
                    "candidate_direction"
                ] = detected_direction

                track[
                    "direction_count"
                ] = 1


            if (
                track["direction_count"]
                >= DIRECTION_CONFIRM_FRAMES
            ):

                track[
                    "direction"
                ] = detected_direction


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            status = (
                "MOVING"
                if speed >= STABLE_SPEED_THRESHOLD
                else
                "STABLE"
            )


            # ------------------------------------------------
            # UPDATE
            # ------------------------------------------------

            track[
                "previous_center"
            ] = old_center

            track[
                "raw_center"
            ] = center

            track[
                "center"
            ] = new_center

            track[
                "velocity"
            ] = (dx, dy)

            track[
                "speed"
            ] = speed

            track[
                "max_speed"
            ] = max(
                track["max_speed"],
                speed
            )

            track[
                "total_distance"
            ] += movement

            track[
                "history"
            ].append(
                new_center
            )

            track[
                "status"
            ] = status

            track[
                "bbox"
            ] = detection["bbox"]

            track[
                "confidence"
            ] = detection.get("confidence", track.get("confidence", 0.0))

            track[
                "last_seen"
            ] = current_time

            track[
                "missing_frames"
            ] = 0

            matched_ids.add(
                best_id
            )


            # ------------------------------------------------
            # AVG SPEED
            # ------------------------------------------------

            if len(
                track["history"]
            ) >= 2:

                points = list(
                    track["history"]
                )

                distances = []

                for i in range(
                    1,
                    len(points)
                ):

                    distances.append(
                        self.distance(
                            points[i - 1],
                            points[i]
                        )
                    )

                if distances:

                    track[
                        "avg_speed"
                    ] = (
                        sum(distances)
                        /
                        len(distances)
                    )


            # ------------------------------------------------
            # EMOTION
            # ------------------------------------------------

            x, y, w, h = (
                detection["bbox"]
            )

            face_crop = frame[
                y:y + h,
                x:x + w
            ]

            if (
                face_crop is not None
                and
                face_crop.size > 0
            ):

                emotion, confidence = (
                    self.detect_emotion(
                        face_crop
                    )
                )

                if emotion != "UNKNOWN":

                    track[
                        "emotion_history"
                    ].append(
                        emotion
                    )

                    track[
                        "confidence_history"
                    ].append(
                        confidence
                    )

                    counter = Counter(
                        track[
                            "emotion_history"
                        ]
                    )

                    track[
                        "emotion"
                    ] = counter.most_common(
                        1
                    )[0][0]

                    track[
                        "emotion_confidence"
                    ] = float(
                        np.mean(
                            track[
                                "confidence_history"
                            ]
                        )
                    )


        # ----------------------------------------------------
        # MISSING FACE
        # ----------------------------------------------------

        remove_ids = []

        for face_id, track in (
            self.face_tracks.items()
        ):

            if face_id not in matched_ids:

                track[
                    "missing_frames"
                ] += 1

                if (
                    track["missing_frames"]
                    >
                    MAX_FACE_MISSING_FRAMES
                ):

                    remove_ids.append(
                        face_id
                    )


        for face_id in remove_ids:

            del self.face_tracks[
                face_id
            ]

        return self.face_tracks


    # ========================================================
    # DRAW FACES
    # ========================================================

    # ========================================================
    # DRAW FACES
    # ========================================================

    def draw_faces(self, frame, tracks):

        box_thickness = self.box_thickness
        use_corners = self.effect == "high"
        label_scale = (0.36 if self.effect == "low" else 0.42) * self.text_scale
        color = self.face_color

        for face_id, face in tracks.items():

            x, y, w, h = face["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            self.draw_reference_box(frame, x1, y1, x2, y2, color, box_thickness, use_corners)

            confidence = float(face.get("confidence", 0.0)) * 100.0
            label = f"FACE-{face_id:02d} {confidence:.0f}%" if confidence > 0 else f"FACE-{face_id:02d}"

            self.draw_reference_label(frame, label, x1, max(20, y1), color, label_scale, self.text_color)

    def detect_objects(
        self,
        frame
    ):

        results = self.yolo_model.predict(
            frame,
            verbose=False,
            conf=OBJECT_CONFIDENCE
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            cls_id = int(
                box.cls[0]
            )

            label = names[cls_id]

            if label.lower() == "person":
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            detections.append({

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": (
                    cx,
                    cy
                ),

                "label": label,

                "confidence": confidence
            })

        return detections


    # ========================================================
    # UPDATE OBJECTS
    # ========================================================

    def update_object_tracks(
        self,
        detections
    ):

        matched_ids = set()

        for detection in detections:

            center = detection[
                "center"
            ]

            label = detection[
                "label"
            ]

            best_id = None

            best_distance = (
                MAX_OBJECT_DISTANCE
            )

            for object_id, track in (
                self.object_tracks.items()
            ):

                if object_id in matched_ids:
                    continue

                if track["label"] != label:
                    continue

                d = self.distance(
                    center,
                    track["center"]
                )

                if d < best_distance:

                    best_distance = d
                    best_id = object_id


            # ------------------------------------------------
            # NEW OBJECT
            # ------------------------------------------------

            if best_id is None:

                object_id = (
                    self.next_object_id
                )

                self.next_object_id += 1

                self.object_tracks[
                    object_id
                ] = {

                    "id": object_id,

                    "bbox": detection["bbox"],

                    "center": center,

                    "previous_center": center,

                    "label": label,

                    "confidence": detection[
                        "confidence"
                    ],

                    "speed": 0.0,

                    "direction": "STABLE",

                    "status": "STABLE",

                    "total_distance": 0.0,

                    "missing_frames": 0
                }

                matched_ids.add(
                    object_id
                )

                continue


            # ------------------------------------------------
            # EXISTING OBJECT
            # ------------------------------------------------

            track = self.object_tracks[
                best_id
            ]

            old_center = track[
                "center"
            ]

            dx = (
                center[0]
                -
                old_center[0]
            )

            dy = (
                center[1]
                -
                old_center[1]
            )

            movement = math.sqrt(
                dx * dx +
                dy * dy
            )

            speed = movement

            if movement < MOVEMENT_THRESHOLD:

                direction = "STABLE"

            elif abs(dx) > abs(dy):

                direction = (
                    "RIGHT"
                    if dx > 0
                    else
                    "LEFT"
                )

            else:

                direction = (
                    "DOWN"
                    if dy > 0
                    else
                    "UP"
                )

            status = (
                "MOVING"
                if speed >= STABLE_SPEED_THRESHOLD
                else
                "STABLE"
            )

            track[
                "previous_center"
            ] = old_center

            track[
                "center"
            ] = center

            track[
                "bbox"
            ] = detection["bbox"]

            track[
                "confidence"
            ] = detection["confidence"]

            track[
                "speed"
            ] = speed

            track[
                "direction"
            ] = direction

            track[
                "status"
            ] = status

            track[
                "total_distance"
            ] += movement

            track[
                "missing_frames"
            ] = 0

            matched_ids.add(
                best_id
            )


        # ----------------------------------------------------
        # MISSING OBJECT
        # ----------------------------------------------------

        remove_ids = []

        for object_id, track in (
            self.object_tracks.items()
        ):

            if object_id not in matched_ids:

                track[
                    "missing_frames"
                ] += 1

                if (
                    track["missing_frames"]
                    >
                    MAX_OBJECT_MISSING_FRAMES
                ):

                    remove_ids.append(
                        object_id
                    )


        for object_id in remove_ids:

            del self.object_tracks[
                object_id
            ]

        return self.object_tracks


    # ========================================================
    # HAND DETECTION
    # ========================================================

    def detect_hands(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = (
            self.hands_detector.process(
                rgb
            )
        )

        detections = []

        if not result.multi_hand_landmarks:

            return detections

        height, width = frame.shape[:2]

        for index, hand_landmarks in enumerate(
            result.multi_hand_landmarks
        ):

            points = []

            xs = []
            ys = []

            for landmark in (
                hand_landmarks.landmark
            ):

                px = int(
                    landmark.x * width
                )

                py = int(
                    landmark.y * height
                )

                px = self.clamp(
                    px,
                    0,
                    width - 1
                )

                py = self.clamp(
                    py,
                    0,
                    height - 1
                )

                points.append(
                    (px, py)
                )

                xs.append(px)
                ys.append(py)


            if not points:
                continue


            cx = int(
                np.mean(xs)
            )

            cy = int(
                np.mean(ys)
            )


            x1 = max(
                0,
                min(xs) - 10
            )

            y1 = max(
                0,
                min(ys) - 10
            )

            x2 = min(
                width - 1,
                max(xs) + 10
            )

            y2 = min(
                height - 1,
                max(ys) + 10
            )


            handedness = "HAND"

            if (
                result.multi_handedness
                and
                index
                <
                len(
                    result.multi_handedness
                )
            ):

                handedness = (
                    result
                    .multi_handedness[index]
                    .classification[0]
                    .label
                )


            detections.append({

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": (
                    cx,
                    cy
                ),

                "landmarks": points,

                "handedness": handedness
            })

        return detections


    # ========================================================
    # UPDATE HAND TRACKS
    # ========================================================

    def update_hand_tracks(
        self,
        detections
    ):

        matched_ids = set()

        for detection in detections:

            center = detection[
                "center"
            ]

            handedness = detection[
                "handedness"
            ]

            best_id = None

            best_distance = (
                MAX_HAND_DISTANCE
            )


            # ------------------------------------------------
            # FIND CLOSEST ACTIVE HAND
            # ------------------------------------------------

            for hand_id, track in (
                self.hand_tracks.items()
            ):

                if track[
                    "missing_frames"
                ] > 0:

                    continue

                if hand_id in matched_ids:

                    continue

                if (
                    track["handedness"]
                    !=
                    handedness
                ):

                    continue

                d = self.distance(
                    center,
                    track["center"]
                )

                if d < best_distance:

                    best_distance = d
                    best_id = hand_id


            # ------------------------------------------------
            # NEW HAND
            # ------------------------------------------------

            if best_id is None:

                hand_id = (
                    self.next_hand_id
                )

                self.next_hand_id += 1

                self.hand_tracks[
                    hand_id
                ] = {

                    "id": hand_id,

                    "bbox": detection["bbox"],

                    "center": center,

                    "previous_center": center,

                    "landmarks": detection[
                        "landmarks"
                    ],

                    "handedness": handedness,

                    "speed": 0.0,

                    "direction": "STABLE",

                    "status": "STABLE",

                    "missing_frames": 0
                }

                matched_ids.add(
                    hand_id
                )

                continue


            # ------------------------------------------------
            # EXISTING HAND
            # ------------------------------------------------

            track = self.hand_tracks[
                best_id
            ]

            old_center = track[
                "center"
            ]

            dx = (
                center[0]
                -
                old_center[0]
            )

            dy = (
                center[1]
                -
                old_center[1]
            )

            movement = math.sqrt(
                dx * dx +
                dy * dy
            )

            speed = movement


            if movement < MOVEMENT_THRESHOLD:

                direction = "STABLE"

            elif abs(dx) > abs(dy):

                direction = (
                    "RIGHT"
                    if dx > 0
                    else
                    "LEFT"
                )

            else:

                direction = (
                    "DOWN"
                    if dy > 0
                    else
                    "UP"
                )


            status = (
                "MOVING"
                if speed >= STABLE_SPEED_THRESHOLD
                else
                "STABLE"
            )


            track[
                "previous_center"
            ] = old_center

            track[
                "center"
            ] = center

            track[
                "bbox"
            ] = detection["bbox"]

            track[
                "landmarks"
            ] = detection["landmarks"]

            track[
                "handedness"
            ] = handedness

            track[
                "speed"
            ] = speed

            track[
                "direction"
            ] = direction

            track[
                "status"
            ] = status

            track[
                "missing_frames"
            ] = 0

            matched_ids.add(
                best_id
            )


        # ----------------------------------------------------
        # MISSING HAND
        # ----------------------------------------------------

        remove_ids = []

        for hand_id, track in (
            self.hand_tracks.items()
        ):

            if hand_id not in matched_ids:

                track[
                    "missing_frames"
                ] += 1

                if (
                    track["missing_frames"]
                    >
                    MAX_HAND_MISSING_FRAMES
                ):

                    remove_ids.append(
                        hand_id
                    )


        for hand_id in remove_ids:

            del self.hand_tracks[
                hand_id
            ]

        return self.hand_tracks


    # ========================================================
    # FIND HOLDING HAND
    # ========================================================

    def find_holding_hand(
        self,
        object_bbox
    ):

        ox1, oy1, ox2, oy2 = (
            object_bbox
        )

        object_center = (
            (ox1 + ox2) / 2,
            (oy1 + oy2) / 2
        )

        object_width = (
            ox2 - ox1
        )

        object_height = (
            oy2 - oy1
        )

        extra_x = (
            object_width * 0.25
        )

        extra_y = (
            object_height * 0.25
        )

        expanded_x1 = (
            ox1 - extra_x
        )

        expanded_y1 = (
            oy1 - extra_y
        )

        expanded_x2 = (
            ox2 + extra_x
        )

        expanded_y2 = (
            oy2 + extra_y
        )

        best_hand = None

        best_distance = float(
            "inf"
        )

        for hand_id, hand in (
            self.hand_tracks.items()
        ):

            if hand[
                "missing_frames"
            ] > 0:

                continue

            cx, cy = hand[
                "center"
            ]

            inside = (
                expanded_x1 <= cx <= expanded_x2
                and
                expanded_y1 <= cy <= expanded_y2
            )

            d = self.distance(
                (cx, cy),
                object_center
            )

            if (
                inside
                and
                d < best_distance
            ):

                best_distance = d

                best_hand = hand_id

        return best_hand


    # ========================================================
    # DRAW OBJECTS
    # ========================================================

    # ========================================================
    # DRAW OBJECTS
    # ========================================================

    def draw_objects(self, frame, tracks):

        box_thickness = self.box_thickness
        use_corners = self.effect == "high"
        label_scale = (0.36 if self.effect == "low" else 0.42) * self.text_scale
        color = self.object_color

        for object_id, obj in tracks.items():

            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]

            self.draw_reference_box(frame, x1, y1, x2, y2, color, box_thickness, use_corners)

            confidence = float(obj.get("confidence", 0.0)) * 100.0
            label_name = str(obj.get("label", "object")).lower()
            label = f"{label_name} {confidence:.0f}%" if confidence > 0 else label_name

            self.draw_reference_label(frame, label, x1, max(20, y1), color, label_scale, self.text_color)

            if self.effect == "high":
                holding_hand = self.find_holding_hand(obj["bbox"])
                if holding_hand is not None:
                    self.draw_reference_label(frame, f"HAND-{holding_hand:02d}", x1, min(frame.shape[0] - 4, y2 + 18), self.hand_color, 0.34 * self.text_scale, self.text_color)

    # ========================================================
    # DRAW HANDS
    # ========================================================

    def draw_hands(self, frame, tracks):

        skeleton_thickness = self.line_thickness
        label_scale = (0.36 if self.effect == "low" else 0.42) * self.text_scale
        color = self.hand_color

        for hand_id, hand in tracks.items():

            if hand.get("missing_frames", 0) > 0:
                continue

            landmarks = hand.get("landmarks")
            if not landmarks:
                continue

            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, skeleton_thickness, cv2.LINE_AA)

            for index, point in enumerate(landmarks):
                radius = 3 if index in (4, 8, 12, 16, 20) else 2
                cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1, cv2.LINE_AA)

            x1, y1, x2, y2 = [int(v) for v in hand["bbox"]]
            self.draw_reference_box(frame, x1, y1, x2, y2, color, self.box_thickness, self.effect == "high")
            self.draw_reference_label(frame, f"HAND-{hand_id:02d}", x1, max(20, y1), color, label_scale, self.text_color)

    # ========================================================
    # HUD
    # ========================================================

    def draw_hud(self, frame, fps, face_count, hand_count, object_count):

        # Keep the video clean; the web UI already displays the counters.
        return

    def process(
        self,
        frame,
        mirror=False,
        effect=None,
        line_size=None,
        box_size=None,
        text_size=None,
        face_color=None,
        hand_color=None,
        object_color=None,
        text_color=None,
    ):

        if frame is None:

            return None

        if effect is not None:
            self.set_effect(effect)

        self.set_style(
            face_color=face_color,
            hand_color=hand_color,
            object_color=object_color,
            text_color=text_color,
            text_size=text_size,
            box_size=box_size,
            line_size=line_size,
        )

        # ----------------------------------------------------
        # MIRROR
        # ----------------------------------------------------

        if mirror:

            frame = cv2.flip(
                frame,
                1
            )


        self.frame_counter += 1


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        delta_time = (
            current_time
            -
            self.previous_time
        )

        if delta_time > 0:

            current_fps = (
                1.0 /
                delta_time
            )

            self.fps = (
                self.fps * 0.90
                +
                current_fps * 0.10
            )

        self.previous_time = (
            current_time
        )


        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        face_detections = (
            self.detect_faces(
                frame
            )
        )

        current_face_tracks = (
            self.update_face_tracks(
                face_detections,
                frame
            )
        )


        # ----------------------------------------------------
        # HAND
        # ----------------------------------------------------

        hand_detections = (
            self.detect_hands(
                frame
            )
        )

        current_hand_tracks = (
            self.update_hand_tracks(
                hand_detections
            )
        )


        # ----------------------------------------------------
        # OBJECT
        # ----------------------------------------------------

        if (
            self.frame_counter
            %
            YOLO_INTERVAL
            ==
            0
        ):

            object_detections = (
                self.detect_objects(
                    frame
                )
            )

            current_object_tracks = (
                self.update_object_tracks(
                    object_detections
                )
            )

        else:

            current_object_tracks = (
                self.object_tracks
            )


        # ----------------------------------------------------
        # DRAW OBJECT
        # ----------------------------------------------------

        self.draw_objects(
            frame,
            current_object_tracks
        )


        # ----------------------------------------------------
        # DRAW HAND
        # ----------------------------------------------------

        self.draw_hands(
            frame,
            current_hand_tracks
        )


        # ----------------------------------------------------
        # DRAW FACE
        # ----------------------------------------------------

        self.draw_faces(
            frame,
            current_face_tracks
        )


        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        self.draw_hud(
            frame,
            self.fps,
            len(current_face_tracks),
            len(current_hand_tracks),
            len(current_object_tracks)
        )


        return frame


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.hands_detector.close()

        except Exception:

            pass

        print(
            "Tracking engine closed."
        )

        