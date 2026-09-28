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

        # ----------------------------------------------------
        # VISUAL SETTINGS
        # ----------------------------------------------------
        # 1.0 = 100%
        self.visual = {
            "line": 1.0,
            "box": 1.0,
            "text": 1.0,
            "marker": 1.0,
        }

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


    def set_visual_settings(self, line=1.0, box=1.0, text=1.0, marker=1.0):
        """Set realtime visual scaling for tracking graphics."""
        try:
            self.visual["line"] = self.clamp(float(line), 0.25, 2.0)
            self.visual["box"] = self.clamp(float(box), 0.25, 2.0)
            self.visual["text"] = self.clamp(float(text), 0.50, 1.75)
            self.visual["marker"] = self.clamp(float(marker), 0.25, 2.0)
        except (TypeError, ValueError):
            pass

    def _line_thickness(self, base=1):
        return max(1, int(round(base * self.visual["line"])))

    def _box_thickness(self, base=1):
        return max(1, int(round(base * self.visual["box"])))

    def _text_scale(self, base=0.45):
        return max(0.25, base * self.visual["text"])

    def _text_thickness(self, base=1):
        return max(1, int(round(base * self.visual["text"])))

    def _marker_size(self, base=8):
        return max(2, int(round(base * self.visual["marker"])))

    def _marker_thickness(self, base=2):
        return max(1, int(round(base * self.visual["marker"])))

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
            "visual": {
                "line": round(self.visual["line"] * 100),
                "box": round(self.visual["box"] * 100),
                "text": round(self.visual["text"] * 100),
                "marker": round(self.visual["marker"] * 100),
            },
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

            results.append({

                "bbox": (
                    x,
                    y,
                    w,
                    h
                ),

                "center": center
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

    def draw_faces(
        self,
        frame,
        tracks
    ):

        current_time = time.time()

        for face_id, face in tracks.items():

            x, y, w, h = face["bbox"]

            cx = int(
                face["center"][0]
            )

            cy = int(
                face["center"][1]
            )

            box_color = (
                LIME
                if face["status"] == "MOVING"
                else
                CYAN,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                self._box_thickness(1),
                cv2.LINE_AA
            )

            self.draw_plus(
                frame,
                (cx, cy),
                box_color,
                self._marker_size(8),
                self._marker_thickness(2)
            )

            if self.effect == "high":
                corner = 14
                thick = self._line_thickness(2)
                cv2.line(frame, (x, y), (x + corner, y), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x, y), (x, y + corner), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x + w, y), (x + w - corner, y), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x + w, y), (x + w, y + corner), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x, y + h), (x + corner, y + h), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x, y + h), (x, y + h - corner), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x + w, y + h), (x + w - corner, y + h), box_color, thick, cv2.LINE_AA)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - corner), box_color, thick, cv2.LINE_AA)

            if self.effect == "low":
                self.draw_text(
                    frame,
                    f"FACE-{face_id:02d}",
                    (x, max(18, y - 8)),
                    box_color,
                    self._text_scale(0.45),
                    self._text_thickness(1)
                )
                continue

            tracking_time = (
                current_time
                -
                face["created_at"]
            )

            panel_width = 245
            panel_height = 210

            panel_x = x + w + 12
            panel_y = y

            if (
                panel_x
                +
                panel_width
                >
                frame.shape[1]
            ):

                panel_x = (
                    x
                    -
                    panel_width
                    -
                    12
                )

            if (
                panel_y
                +
                panel_height
                >
                frame.shape[0]
            ):

                panel_y = max(
                    5,
                    frame.shape[0]
                    -
                    panel_height
                    -
                    5
                )

            self.draw_panel(
                frame,
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                box_color
            )

            self.draw_text(
                frame,
                f"FACE-{face_id:02d}",
                (
                    panel_x + 10,
                    panel_y + 20
                ),
                box_color,
                self._text_scale(0.55),
                self._text_thickness(2)
            )

            self.draw_text(
                frame,
                f"X {cx:04d}   Y {cy:04d}",
                (
                    panel_x + 10,
                    panel_y + 43
                ),
                WHITE,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"DIR  {face['direction']}",
                (
                    panel_x + 10,
                    panel_y + 64
                ),
                BLUE,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"SPEED {face['speed']:5.1f}",
                (
                    panel_x + 10,
                    panel_y + 84
                ),
                LIME,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"AVG   {face['avg_speed']:5.1f}",
                (
                    panel_x + 10,
                    panel_y + 103
                ),
                CYAN,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"MAX   {face['max_speed']:5.1f}",
                (
                    panel_x + 10,
                    panel_y + 122
                ),
                AMBER,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            status_color = (
                LIME
                if face["status"] == "MOVING"
                else
                GRAY,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"STATUS {face['status']}",
                (
                    panel_x + 10,
                    panel_y + 141
                ),
                status_color,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"EXP   {face['emotion']}",
                (
                    panel_x + 10,
                    panel_y + 161
                ),
                MAGENTA,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            confidence = (
                face["emotion_confidence"]
                *
                100
            )

            self.draw_text(
                frame,
                f"CONF  {confidence:4.1f}%",
                (
                    panel_x + 10,
                    panel_y + 180
                ),
                WHITE,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"TIME  {tracking_time:5.1f}s",
                (
                    panel_x + 10,
                    panel_y + 199
                ),
                GRAY,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            self.draw_text(
                frame,
                f"DIST {face['total_distance']:6.1f}",
                (
                    panel_x + 130,
                    panel_y + 199
                ),
                AMBER,
                self._text_scale(0.45),
                self._text_thickness(1)
            )


    # ========================================================
    # OBJECT DETECTION
    # ========================================================

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

    def draw_objects(
        self,
        frame,
        tracks
    ):

        for object_id, obj in (
            tracks.items()
        ):

            x1, y1, x2, y2 = (
                obj["bbox"]
            )

            cx = int(
                obj["center"][0]
            )

            cy = int(
                obj["center"][1]
            )

            box_color = (
                AMBER
                if obj["status"] == "MOVING"
                else
                BLUE
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                self._box_thickness(1),
                cv2.LINE_AA
            )

            self.draw_plus(
                frame,
                (cx, cy),
                box_color,
                self._marker_size(6),
                self._marker_thickness(1)
            )

            if self.effect == "high":
                corner = 10
                cv2.line(frame, (x1, y1), (x1 + corner, y1), box_color, self._line_thickness(2), cv2.LINE_AA)
                cv2.line(frame, (x1, y1), (x1, y1 + corner), box_color, self._line_thickness(2), cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2 - corner, y2), box_color, self._line_thickness(2), cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2, y2 - corner), box_color, self._line_thickness(2), cv2.LINE_AA)

            if self.effect == "low":
                self.draw_text(
                    frame,
                    f"OBJ-{object_id:02d} {obj['label'].upper()}",
                    (x1, max(18, y1 - 8)),
                    box_color,
                    self._text_scale(0.42),
                    self._text_thickness(1)
                )
                continue


            holding_hand = (
                self.find_holding_hand(
                    obj["bbox"]
                )
            )

            if holding_hand is not None:

                holding_text = (
                    f"HELD • HAND-{holding_hand:02d}"
                )

            else:

                holding_text = ""


            label = (
                f"OBJ-{object_id:02d} "
                f"{obj['label'].upper()}"
            )

            confidence = (
                obj["confidence"] * 100
            )

            panel_width = 220

            panel_height = (
                92
                if holding_hand is not None
                else
                72
            )

            panel_x = x1

            panel_y = (
                y1
                -
                panel_height
                -
                8
            )

            if panel_y < 5:

                panel_y = y2 + 8

            if (
                panel_x
                +
                panel_width
                >
                frame.shape[1]
            ):

                panel_x = (
                    frame.shape[1]
                    -
                    panel_width
                    -
                    5
                )


            self.draw_panel(
                frame,
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                box_color
            )

            self.draw_text(
                frame,
                label,
                (
                    panel_x + 8,
                    panel_y + 19
                ),
                box_color,
                self._text_scale(0.48),
                self._text_thickness(2)
            )

            self.draw_text(
                frame,
                f"CONF {confidence:4.1f}%",
                (
                    panel_x + 8,
                    panel_y + 39
                ),
                WHITE,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            status_color = (
                LIME
                if obj["status"] == "MOVING"
                else
                GRAY
            )

            self.draw_text(
                frame,
                (
                    f"{obj['direction']} "
                    f"{obj['speed']:4.1f}"
                ),
                (
                    panel_x + 8,
                    panel_y + 58
                ),
                status_color,
                self._text_scale(0.45),
                self._text_thickness(1)
            )

            if holding_hand is not None:

                self.draw_text(
                    frame,
                    holding_text,
                    (
                        panel_x + 8,
                        panel_y + 79
                    ),
                    MAGENTA,
                    self._text_scale(0.42),
                    self._text_thickness(1)
                )


    # ========================================================
    # DRAW HANDS
    # ========================================================

    def draw_hands(
        self,
        frame,
        tracks
    ):

        for hand_id, hand in (
            tracks.items()
        ):

            # ----------------------------------------------
            # GHOST PREVENTION
            # ----------------------------------------------

            if hand[
                "missing_frames"
            ] > 0:

                continue

            landmarks = hand.get(
                "landmarks",
                None
            )

            if landmarks is None:

                continue


            cx = int(
                hand["center"][0]
            )

            cy = int(
                hand["center"][1]
            )


            if hand["status"] == "MOVING":

                hand_color = MAGENTA

            else:

                hand_color = CYAN


            # ----------------------------------------------
            # SKELETON
            # ----------------------------------------------

            for start_idx, end_idx in (
                HAND_CONNECTIONS
            ):

                if (
                    start_idx >= len(landmarks)
                    or
                    end_idx >= len(landmarks)
                ):

                    continue

                p1 = landmarks[
                    start_idx
                ]

                p2 = landmarks[
                    end_idx
                ]

                cv2.line(
                    frame,
                    (
                        int(p1[0]),
                        int(p1[1])
                    ),
                    (
                        int(p2[0]),
                        int(p2[1])
                    ),
                    hand_color,
                    self._line_thickness(2 if self.effect == "high" else 1),
                    cv2.LINE_AA
                )


            # ----------------------------------------------
            # LANDMARK POINTS
            # ----------------------------------------------

            for index, point in enumerate(
                landmarks
            ):

                px = int(point[0])
                py = int(point[1])

                if index in [
                    4,
                    8,
                    12,
                    16,
                    20
                ]:

                    radius = self._marker_size(3)

                else:

                    radius = self._marker_size(2)

                cv2.circle(
                    frame,
                    (
                        px,
                        py
                    ),
                    radius,
                    hand_color,
                    -1,
                    cv2.LINE_AA
                )


            # ----------------------------------------------
            # CENTER PLUS
            # ----------------------------------------------

            self.draw_plus(
                frame,
                (cx, cy),
                hand_color,
                self._marker_size(6 if self.effect != "high" else 8),
                self._marker_thickness(1 if self.effect != "high" else 2)
            )

            if self.effect == "low":
                continue


            # ----------------------------------------------
            # LABEL
            # ----------------------------------------------

            label_x = cx + 12
            label_y = cy - 12

            if (
                label_x + 120
                >
                frame.shape[1]
            ):

                label_x = cx - 120

            if label_y < 20:

                label_y = cy + 25


            self.draw_text(
                frame,
                f"HAND-{hand_id:02d}",
                (
                    label_x,
                    label_y
                ),
                hand_color,
                self._text_scale(0.42),
                self._text_thickness(1)
            )


            status_color = (
                LIME
                if hand["status"] == "MOVING"
                else
                GRAY
            )

            self.draw_text(
                frame,
                (
                    f"{hand['direction']} "
                    f"{hand['speed']:4.1f}"
                ),
                (
                    label_x,
                    label_y + 17
                ),
                status_color,
                self._text_scale(0.38),
                self._text_thickness(1)
            )


    # ========================================================
    # HUD
    # ========================================================

    def draw_hud(
        self,
        frame,
        fps,
        face_count,
        hand_count,
        object_count
    ):

        height, width = (
            frame.shape[:2]
        )

        hud_width = 330
        hud_height = 82

        self.draw_panel(
            frame,
            12,
            12,
            hud_width,
            hud_height,
            CYAN,
            0.75
        )

        self.draw_text(
            frame,
            "BLOB TRACKING SYSTEM",
            (
                24,
                34
            ),
            CYAN,
            self._text_scale(0.58),
            self._text_thickness(2)
        )

        self.draw_text(
            frame,
            f"FPS {fps:4.1f}",
            (
                24,
                55
            ),
            WHITE,
            self._text_scale(0.45),
            self._text_thickness(1)
        )

        self.draw_text(
            frame,
            f"FACE {face_count:02d}",
            (
                105,
                55
            ),
            LIME,
            self._text_scale(0.45),
            self._text_thickness(1)
        )

        self.draw_text(
            frame,
            f"HAND {hand_count:02d}",
            (
                190,
                55
            ),
            MAGENTA,
            self._text_scale(0.45),
            self._text_thickness(1)
        )

        self.draw_text(
            frame,
            f"OBJ {object_count:02d}",
            (
                275,
                55
            ),
            AMBER,
            self._text_scale(0.45),
            self._text_thickness(1)
        )


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process(
        self,
        frame,
        mirror=False,
        effect=None,
        line=None,
        box=None,
        text_size=None,
        marker=None
    ):

        if frame is None:

            return None

        if effect is not None:
            self.set_effect(effect)

        if any(value is not None for value in (line, box, text_size, marker)):
            self.set_visual_settings(
                self.visual["line"] if line is None else line,
                self.visual["box"] if box is None else box,
                self.visual["text"] if text_size is None else text_size,
                self.visual["marker"] if marker is None else marker,
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
        