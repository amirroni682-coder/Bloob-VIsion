from flask import (
    Flask,
    render_template,
    request,
    Response,
    jsonify,
    send_file,
)

import atexit
import inspect
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np

from tracking_engine import TrackingEngine


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Maximum request size.
# Default: 1024 MB.
try:
    MAX_UPLOAD_MB = float(
        os.environ.get(
            "MAX_UPLOAD_MB",
            "1024"
        )
    )
except (TypeError, ValueError):
    MAX_UPLOAD_MB = 1024.0

MAX_UPLOAD_MB = max(
    64.0,
    min(MAX_UPLOAD_MB, 4096.0)
)

app.config["MAX_CONTENT_LENGTH"] = int(
    MAX_UPLOAD_MB * 1024 * 1024
)


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg_path():
    """
    Find FFmpeg from:

    1. System PATH
    2. imageio-ffmpeg bundled binary
    """

    # --------------------------------------------------------
    # 1. System PATH
    # --------------------------------------------------------

    system_ffmpeg = shutil.which(
        "ffmpeg"
    )

    if system_ffmpeg:
        print(
            "FFmpeg found in PATH:",
            system_ffmpeg
        )

        return system_ffmpeg

    # --------------------------------------------------------
    # 2. imageio-ffmpeg fallback
    # --------------------------------------------------------

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        if (
            bundled_ffmpeg
            and os.path.exists(
                bundled_ffmpeg
            )
        ):
            print(
                "FFmpeg found via imageio-ffmpeg:",
                bundled_ffmpeg
            )

            return bundled_ffmpeg

    except Exception as exc:
        print(
            "imageio-ffmpeg lookup failed:",
            exc
        )

    return None


FFMPEG_PATH = get_ffmpeg_path()


print("========================================")
print("BLOB VISION")
print("========================================")
print(
    "FFMPEG:",
    FFMPEG_PATH or "NOT FOUND"
)
print(
    "MAX_UPLOAD_MB:",
    MAX_UPLOAD_MB
)
print("========================================")


# ============================================================
# TRACKING ENGINE
# ============================================================

print("Starting Tracking Engine...")

engine = TrackingEngine()

print("Tracking Engine loaded.")


# ============================================================
# THREAD SAFETY
# ============================================================

engine_lock = threading.Lock()

video_jobs = {}

video_jobs_lock = threading.Lock()

full_video_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value,
    default
):
    try:
        return float(value)
    except (
        TypeError,
        ValueError
    ):
        return default


def safe_bool(
    value,
    default=False
):
    if value is None:
        return default

    value = str(value).strip().lower()

    return value in (
        "1",
        "true",
        "yes",
        "on"
    )


def safe_hex_color(
    value,
    default="#64eaff"
):
    """
    Normalize color to #RRGGBB.

    The tracking engine versions used in this project
    may accept colors differently, so this function
    keeps the incoming value clean.
    """

    if not value:
        return default

    value = str(value).strip()

    if not value.startswith("#"):
        value = "#" + value

    if len(value) != 7:
        return default

    allowed = set(
        "0123456789abcdefABCDEF"
    )

    if any(
        char not in allowed
        for char in value[1:]
    ):
        return default

    return value.lower()


def hex_to_bgr(
    color,
    default=(255, 220, 80)
):
    """
    Convert #RRGGBB -> OpenCV BGR tuple.
    """

    color = safe_hex_color(
        color
    )

    try:
        red = int(
            color[1:3],
            16
        )

        green = int(
            color[3:5],
            16
        )

        blue = int(
            color[5:7],
            16
        )

        return (
            blue,
            green,
            red
        )

    except Exception:
        return default


def parse_tracking_settings(
    source
):
    """
    Read tracking settings from either:

    - HTTP headers
    - Flask request.form
    """

    if source is None:
        source = {}

    # --------------------------------------------------------
    # EFFECT
    # --------------------------------------------------------

    effect = (
        source.get(
            "X-Tracking-Effect",
            source.get(
                "effect",
                "medium"
            )
        )
        or "medium"
    )

    effect = str(
        effect
    ).lower()

    if effect not in (
        "low",
        "medium",
        "high"
    ):
        effect = "medium"

    # --------------------------------------------------------
    # LINE SIZE
    # --------------------------------------------------------

    line_size = safe_float(
        source.get(
            "X-Line-Size",
            source.get(
                "line_size",
                2.0
            )
        ),
        2.0
    )

    # --------------------------------------------------------
    # BOX SIZE
    # --------------------------------------------------------

    box_size = safe_float(
        source.get(
            "X-Box-Size",
            source.get(
                "box_size",
                2.0
            )
        ),
        2.0
    )

    # --------------------------------------------------------
    # TEXT SIZE
    # --------------------------------------------------------

    text_size = safe_float(
        source.get(
            "X-Text-Size",
            source.get(
                "text_size",
                0.45
            )
        ),
        0.45
    )

    line_size = max(
        1.0,
        min(
            line_size,
            8.0
        )
    )

    box_size = max(
        1.0,
        min(
            box_size,
            8.0
        )
    )

    text_size = max(
        0.25,
        min(
            text_size,
            1.0
        )
    )

    # --------------------------------------------------------
    # MIRROR
    # --------------------------------------------------------

    mirror_value = source.get(
        "X-Mirror",
        source.get(
            "mirror",
            "true"
        )
    )

    mirror = safe_bool(
        mirror_value,
        default=True
    )

    # --------------------------------------------------------
    # COLORS
    # --------------------------------------------------------

    face_color = safe_hex_color(
        source.get(
            "X-Face-Color",
            source.get(
                "face_color",
                "#64eaff"
            )
        ),
        "#64eaff"
    )

    hand_color = safe_hex_color(
        source.get(
            "X-Hand-Color",
            source.get(
                "hand_color",
                "#7dff9d"
            )
        ),
        "#7dff9d"
    )

    object_color = safe_hex_color(
        source.get(
            "X-Object-Color",
            source.get(
                "object_color",
                "#ffca72"
            )
        ),
        "#ffca72"
    )

    text_color = safe_hex_color(
        source.get(
            "X-Text-Color",
            source.get(
                "text_color",
                "#ffffff"
            )
        ),
        "#ffffff"
    )

    return {
        "effect": effect,
        "line_size": line_size,
        "box_size": box_size,
        "text_size": text_size,
        "mirror": mirror,
        "face_color": face_color,
        "hand_color": hand_color,
        "object_color": object_color,
        "text_color": text_color,
    }


def apply_engine_colors(
    settings
):
    """
    Compatibility layer for different versions
    of tracking_engine.py.

    If the engine has:
        - set_colors()
        - face_color
        - hand_color
        - object_color
        - text_color

    they will be updated.

    This does NOT require the engine to support
    color parameters directly in process().
    """

    face_bgr = hex_to_bgr(
        settings["face_color"],
        (255, 220, 80)
    )

    hand_bgr = hex_to_bgr(
        settings["hand_color"],
        (120, 255, 120)
    )

    object_bgr = hex_to_bgr(
        settings["object_color"],
        (80, 190, 255)
    )

    text_bgr = hex_to_bgr(
        settings["text_color"],
        (255, 255, 255)
    )

    # --------------------------------------------------------
    # Try generic set_colors()
    # --------------------------------------------------------

    set_colors = getattr(
        engine,
        "set_colors",
        None
    )

    if callable(set_colors):

        attempts = [
            {
                "face_color": face_bgr,
                "hand_color": hand_bgr,
                "object_color": object_bgr,
                "text_color": text_bgr,
            },
            {
                "face_color": settings["face_color"],
                "hand_color": settings["hand_color"],
                "object_color": settings["object_color"],
                "text_color": settings["text_color"],
            },
        ]

        for kwargs in attempts:

            try:
                set_colors(
                    **kwargs
                )

                break

            except Exception:
                pass

    # --------------------------------------------------------
    # Try direct attributes
    # --------------------------------------------------------

    attribute_values = {
        "face_color": face_bgr,
        "hand_color": hand_bgr,
        "object_color": object_bgr,
        "text_color": text_bgr,
    }

    for attribute, value in (
        attribute_values.items()
    ):

        try:

            if hasattr(
                engine,
                attribute
            ):
                setattr(
                    engine,
                    attribute,
                    value
                )

        except Exception:
            pass


def process_engine_frame(
    frame,
    settings
):
    """
    Process a single frame while remaining compatible
    with multiple TrackingEngine.process() signatures.
    """

    if frame is None:
        return None

    # --------------------------------------------------------
    # Apply colors through compatibility layer
    # --------------------------------------------------------

    apply_engine_colors(
        settings
    )

    # --------------------------------------------------------
    # Inspect engine.process()
    # --------------------------------------------------------

    try:

        parameters = inspect.signature(
            engine.process
        ).parameters

    except Exception:

        parameters = {}

    kwargs = {
        "mirror": settings["mirror"],
        "effect": settings["effect"],
        "line_size": settings["line_size"],
        "box_size": settings["box_size"],
        "text_size": settings["text_size"],
    }

    # --------------------------------------------------------
    # Newer engine versions may support colors directly
    # --------------------------------------------------------

    color_values = {
        "face_color": settings["face_color"],
        "hand_color": settings["hand_color"],
        "object_color": settings["object_color"],
        "text_color": settings["text_color"],
    }

    for name, value in (
        color_values.items()
    ):

        if name in parameters:

            kwargs[name] = value

    # --------------------------------------------------------
    # Only send supported arguments.
    # --------------------------------------------------------

    if parameters:

        supports_kwargs = any(
            parameter.kind
            == inspect.Parameter.VAR_KEYWORD
            for parameter
            in parameters.values()
        )

        if not supports_kwargs:

            kwargs = {
                key: value
                for key, value
                in kwargs.items()
                if key in parameters
            }

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    return engine.process(
        frame,
        **kwargs
    )


def rotate_frame(
    frame,
    rotation
):
    """
    Apply orientation rotation.

    OpenCV:
        90  = clockwise
        180 = upside down
        270 = counter-clockwise
    """

    if frame is None:
        return frame

    rotation = int(
        rotation or 0
    )

    if rotation == 90:

        return cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )

    if rotation == 180:

        return cv2.rotate(
            frame,
            cv2.ROTATE_180
        )

    if rotation == 270:

        return cv2.rotate(
            frame,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return frame


def cleanup_file(
    path
):
    if not path:
        return

    try:

        if os.path.exists(path):
            os.remove(path)

    except Exception as exc:

        print(
            "Cleanup file error:",
            exc
        )


def cleanup_directory(
    path
):
    if not path:
        return

    try:

        if os.path.isdir(path):
            shutil.rmtree(
                path,
                ignore_errors=True
            )

    except Exception as exc:

        print(
            "Cleanup directory error:",
            exc
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/healthz",
    methods=["GET"]
)
def healthz():

    return jsonify({
        "status": "ok",
        "service": "blob-vision",
        "ffmpeg": bool(
            FFMPEG_PATH
        )
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PROCESS FRAME
# ============================================================

@app.route(
    "/process_frame",
    methods=["POST"]
)
def process_frame():

    data = request.get_data()

    if not data:

        return Response(
            "No frame received",
            status=400
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return Response(
            "Invalid frame",
            status=400
        )

    settings = parse_tracking_settings(
        request.headers
    )

    try:

        with engine_lock:

            processed = process_engine_frame(
                frame,
                settings
            )

        if processed is None:

            return Response(
                "Processing failed",
                status=500
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            processed,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                80
            ]
        )

        if not ok:

            return Response(
                "JPEG encoding failed",
                status=500
            )

        return Response(
            encoded.tobytes(),
            mimetype="image/jpeg"
        )

    except Exception as exc:

        print(
            "PROCESS FRAME ERROR:",
            exc
        )

        return Response(
            "Processing error",
            status=500
        )


# ============================================================
# STATS
# ============================================================

@app.route(
    "/stats",
    methods=["GET"]
)
def stats():

    try:

        with engine_lock:

            result = engine.get_stats()

        if not isinstance(
            result,
            dict
        ):
            result = {}

        return jsonify(
            result
        )

    except Exception as exc:

        print(
            "STATS ERROR:",
            exc
        )

        return jsonify({
            "fps": 0,
            "face": 0,
            "hand": 0,
            "object": 0,
            "effect": "medium",
            "error": str(exc)
        })


# ============================================================
# RESET TRACKING
# ============================================================

@app.route(
    "/reset_tracking",
    methods=["POST"]
)
def reset_tracking():

    try:

        with engine_lock:

            engine.reset()

        return jsonify({
            "success": True
        })

    except Exception as exc:

        print(
            "RESET ERROR:",
            exc
        )

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


# ============================================================
# WEBM -> MP4
# ============================================================

@app.route(
    "/convert_to_mp4",
    methods=["POST"]
)
def convert_to_mp4():

    input_path = None
    output_path = None
    cleanup_after_send = False

    try:

        # ----------------------------------------------------
        # FFmpeg check
        # ----------------------------------------------------

        if not FFMPEG_PATH:

            return jsonify({
                "success": False,
                "error": (
                    "FFmpeg tidak ditemukan."
                )
            }), 500

        if not os.path.exists(
            FFMPEG_PATH
        ):

            return jsonify({
                "success": False,
                "error": (
                    "Path FFmpeg tidak valid."
                )
            }), 500

        # ----------------------------------------------------
        # File
        # ----------------------------------------------------

        uploaded_file = request.files.get(
            "video"
        )

        if uploaded_file is None:

            return jsonify({
                "success": False,
                "error": (
                    "Video file tidak ditemukan."
                )
            }), 400

        if not uploaded_file.filename:

            return jsonify({
                "success": False,
                "error": (
                    "Nama file kosong."
                )
            }), 400

        # ----------------------------------------------------
        # Temporary files
        # ----------------------------------------------------

        input_temp = tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False
        )

        input_path = (
            input_temp.name
        )

        uploaded_file.save(
            input_path
        )

        input_temp.close()

        output_temp = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )

        output_path = (
            output_temp.name
        )

        output_temp.close()

        # ----------------------------------------------------
        # FFmpeg
        # ----------------------------------------------------

        command = [
            FFMPEG_PATH,

            "-y",

            "-i",
            input_path,

            "-map",
            "0:v:0",

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "23",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-map_metadata",
            "-1",

            "-an",

            output_path
        ]

        print(
            "Starting WebM -> MP4 conversion..."
        )

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:

            print(
                "FFMPEG ERROR:"
            )

            print(
                result.stderr
            )

            return jsonify({
                "success": False,
                "error": (
                    "FFmpeg gagal mengubah "
                    "WebM menjadi MP4."
                )
            }), 500

        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        if not os.path.exists(
            output_path
        ):

            return jsonify({
                "success": False,
                "error": (
                    "File MP4 tidak berhasil dibuat."
                )
            }), 500

        output_size = os.path.getsize(
            output_path
        )

        if output_size <= 0:

            return jsonify({
                "success": False,
                "error": (
                    "File MP4 kosong."
                )
            }), 500

        print(
            "MP4 conversion successful:",
            output_size,
            "bytes"
        )

        response = send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="blob_tracking.mp4"
        )

        cleanup_after_send = True

        @response.call_on_close
        def cleanup_files():

            cleanup_file(
                input_path
            )

            cleanup_file(
                output_path
            )

        return response

    except Exception as exc:

        print(
            "MP4 CONVERSION ERROR:",
            exc
        )

        cleanup_file(
            input_path
        )

        cleanup_file(
            output_path
        )

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


# ============================================================
# FULL VIDEO PROCESSING
# ============================================================

def process_full_video_job(
    job_id,
    input_path,
    job_dir,
    settings
):
    """
    Process complete uploaded video.

    The original video's orientation metadata is checked
    so 9:16 / rotated phone videos keep the proper orientation.
    """

    output_intermediate = None
    output_final = None

    try:

        # ----------------------------------------------------
        # Only one full-video task at a time
        # ----------------------------------------------------

        with full_video_lock:

            # ------------------------------------------------
            # Update job
            # ------------------------------------------------

            with video_jobs_lock:

                job = video_jobs.get(
                    job_id
                )

                if job is None:
                    return

                job["status"] = (
                    "processing"
                )

                job["progress"] = 0

                job["message"] = (
                    "Opening video..."
                )

            # ------------------------------------------------
            # Open input
            # ------------------------------------------------

            capture = cv2.VideoCapture(
                input_path
            )

            if not capture.isOpened():

                raise RuntimeError(
                    "Video tidak bisa dibuka."
                )

            # ------------------------------------------------
            # Read rotation metadata
            # ------------------------------------------------

            rotation = 0

            try:

                rotation_value = capture.get(
                    cv2.CAP_PROP_ORIENTATION_META
                )

                if rotation_value is not None:

                    rotation = int(
                        round(
                            rotation_value
                        )
                    )

            except Exception:

                rotation = 0

            # ------------------------------------------------
            # Disable OpenCV automatic orientation
            # because we apply it ourselves.
            # ------------------------------------------------

            try:

                capture.set(
                    cv2.CAP_PROP_ORIENTATION_AUTO,
                    0
                )

            except Exception:
                pass

            # ------------------------------------------------
            # Source metadata
            # ------------------------------------------------

            source_width = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            source_height = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            total_frames = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            # ------------------------------------------------
            # Safe FPS
            # ------------------------------------------------

            if (
                not fps
                or fps <= 0
                or np.isnan(fps)
            ):
                fps = 30.0

            # ------------------------------------------------
            # Read first frame
            # ------------------------------------------------

            ok, first_frame = capture.read()

            if not ok or first_frame is None:

                capture.release()

                raise RuntimeError(
                    "Frame pertama video "
                    "tidak bisa dibaca."
                )

            # ------------------------------------------------
            # Apply orientation
            # ------------------------------------------------

            first_frame = rotate_frame(
                first_frame,
                rotation
            )

            output_height, output_width = (
                first_frame.shape[:2]
            )

            # ------------------------------------------------
            # Intermediate OpenCV MP4
            # ------------------------------------------------

            output_intermediate = os.path.join(
                job_dir,
                "tracked_intermediate.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_intermediate,
                fourcc,
                float(fps),
                (
                    int(output_width),
                    int(output_height)
                )
            )

            if not writer.isOpened():

                capture.release()

                raise RuntimeError(
                    "VideoWriter gagal dibuka."
                )

            # ------------------------------------------------
            # Reset engine before processing
            # ------------------------------------------------

            with engine_lock:

                try:
                    engine.reset()
                except Exception:
                    pass

            current_frame_number = 0

            # ------------------------------------------------
            # Process frames
            # ------------------------------------------------

            while True:

                if current_frame_number == 0:

                    frame = first_frame

                else:

                    ok, frame = (
                        capture.read()
                    )

                    if (
                        not ok
                        or frame is None
                    ):
                        break

                    frame = rotate_frame(
                        frame,
                        rotation
                    )

                # --------------------------------------------
                # Safety dimension normalization
                # --------------------------------------------

                if (
                    frame.shape[1]
                    != output_width
                    or
                    frame.shape[0]
                    != output_height
                ):

                    frame = cv2.resize(
                        frame,
                        (
                            output_width,
                            output_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                # --------------------------------------------
                # Tracking
                # --------------------------------------------

                with engine_lock:

                    processed = (
                        process_engine_frame(
                            frame,
                            settings
                        )
                    )

                if processed is None:

                    raise RuntimeError(
                        "Tracking engine "
                        "mengembalikan frame kosong."
                    )

                # --------------------------------------------
                # Writer
                # --------------------------------------------

                if (
                    processed.shape[1]
                    != output_width
                    or
                    processed.shape[0]
                    != output_height
                ):

                    processed = cv2.resize(
                        processed,
                        (
                            output_width,
                            output_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                writer.write(
                    processed
                )

                current_frame_number += 1

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                if total_frames > 0:

                    progress = (
                        current_frame_number
                        /
                        total_frames
                    ) * 100.0

                    progress = max(
                        0.0,
                        min(
                            progress,
                            99.0
                        )
                    )

                else:

                    progress = 0.0

                if (
                    current_frame_number % 5 == 0
                    or current_frame_number == 1
                ):

                    with video_jobs_lock:

                        if job_id in video_jobs:

                            video_jobs[
                                job_id
                            ]["progress"] = round(
                                progress,
                                1
                            )

                            video_jobs[
                                job_id
                            ]["message"] = (
                                f"Processing frame "
                                f"{current_frame_number}"
                                + (
                                    f"/{total_frames}"
                                    if total_frames > 0
                                    else ""
                                )
                            )

            # ------------------------------------------------
            # Release
            # ------------------------------------------------

            capture.release()
            writer.release()

            # ------------------------------------------------
            # Validate intermediate
            # ------------------------------------------------

            if not os.path.exists(
                output_intermediate
            ):

                raise RuntimeError(
                    "Output video tidak terbentuk."
                )

            intermediate_size = (
                os.path.getsize(
                    output_intermediate
                )
            )

            if intermediate_size <= 0:

                raise RuntimeError(
                    "Output video kosong."
                )

            # ------------------------------------------------
            # Update status
            # ------------------------------------------------

            with video_jobs_lock:

                if job_id in video_jobs:

                    video_jobs[
                        job_id
                    ]["progress"] = 99

                    video_jobs[
                        job_id
                    ]["message"] = (
                        "Encoding final MP4..."
                    )

            # ------------------------------------------------
            # Final H.264 MP4
            # ------------------------------------------------

            output_final = os.path.join(
                job_dir,
                "blob_vision_processed.mp4"
            )

            if not FFMPEG_PATH:

                raise RuntimeError(
                    "FFmpeg tidak ditemukan."
                )

            command = [
                FFMPEG_PATH,

                "-y",

                "-i",
                output_intermediate,

                "-map",
                "0:v:0",

                "-c:v",
                "libx264",

                "-preset",
                "veryfast",

                "-crf",
                "23",

                "-pix_fmt",
                "yuv420p",

                "-movflags",
                "+faststart",

                # Prevent stale phone rotation
                # metadata from changing playback.
                "-map_metadata",
                "-1",

                "-an",

                output_final
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:

                print(
                    "FINAL FFMPEG ERROR:"
                )

                print(
                    result.stderr
                )

                raise RuntimeError(
                    "FFmpeg gagal melakukan "
                    "final encoding."
                )

            # ------------------------------------------------
            # Validate final output
            # ------------------------------------------------

            if not os.path.exists(
                output_final
            ):

                raise RuntimeError(
                    "File final MP4 tidak ditemukan."
                )

            final_size = os.path.getsize(
                output_final
            )

            if final_size <= 0:

                raise RuntimeError(
                    "File final MP4 kosong."
                )

            # ------------------------------------------------
            # Finished
            # ------------------------------------------------

            with video_jobs_lock:

                if job_id in video_jobs:

                    video_jobs[
                        job_id
                    ]["status"] = (
                        "completed"
                    )

                    video_jobs[
                        job_id
                    ]["progress"] = 100

                    video_jobs[
                        job_id
                    ]["message"] = (
                        "Processing complete."
                    )

                    video_jobs[
                        job_id
                    ]["result_path"] = (
                        output_final
                    )

                    video_jobs[
                        job_id
                    ]["filename"] = (
                        "blob_vision_processed.mp4"
                    )

                    video_jobs[
                        job_id
                    ]["size"] = (
                        final_size
                    )

                    video_jobs[
                        job_id
                    ]["rotation"] = (
                        rotation
                    )

                    video_jobs[
                        job_id
                    ]["width"] = (
                        output_width
                    )

                    video_jobs[
                        job_id
                    ]["height"] = (
                        output_height
                    )

    except Exception as exc:

        print(
            "FULL VIDEO JOB ERROR:",
            job_id,
            exc
        )

        with video_jobs_lock:

            if job_id in video_jobs:

                video_jobs[
                    job_id
                ]["status"] = (
                    "error"
                )

                video_jobs[
                    job_id
                ]["message"] = (
                    "Processing failed."
                )

                video_jobs[
                    job_id
                ]["error"] = str(
                    exc
                )

    finally:

        # ----------------------------------------------------
        # Input file can always be removed
        # ----------------------------------------------------

        cleanup_file(
            input_path
        )

        # ----------------------------------------------------
        # Intermediate can be removed.
        # Keep final output.
        # ----------------------------------------------------

        cleanup_file(
            output_intermediate
        )


# ============================================================
# START FULL VIDEO PROCESSING
# ============================================================

@app.route(
    "/process_full_video",
    methods=["POST"]
)
def process_full_video():

    uploaded_file = request.files.get(
        "video"
    )

    if uploaded_file is None:

        return jsonify({
            "success": False,
            "error": (
                "Video file tidak ditemukan."
            )
        }), 400

    if not uploaded_file.filename:

        return jsonify({
            "success": False,
            "error": (
                "Nama file kosong."
            )
        }), 400

    # --------------------------------------------------------
    # One job = one temporary directory
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex

    job_dir = tempfile.mkdtemp(
        prefix="blobvision_"
    )

    input_path = os.path.join(
        job_dir,
        "input_video"
    )

    # --------------------------------------------------------
    # Determine extension
    # --------------------------------------------------------

    original_name = (
        uploaded_file.filename
        or "input.mp4"
    )

    _, extension = os.path.splitext(
        original_name
    )

    extension = (
        extension.lower()
        if extension
        else ".mp4"
    )

    allowed_extensions = (
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".m4v"
    )

    if extension not in allowed_extensions:

        cleanup_directory(
            job_dir
        )

        return jsonify({
            "success": False,
            "error": (
                "Format video tidak didukung."
            )
        }), 400

    input_path += extension

    try:

        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        uploaded_file.save(
            input_path
        )

        if not os.path.exists(
            input_path
        ):

            raise RuntimeError(
                "Upload file gagal disimpan."
            )

        # ----------------------------------------------------
        # Settings from form
        # ----------------------------------------------------

        settings = (
            parse_tracking_settings(
                request.form
            )
        )

        # Uploaded video must not mirror
        # unless explicitly requested.
        #
        # Frontend normally sends false.
        #
        if (
            "mirror"
            not in request.form
            and
            "X-Mirror"
            not in request.form
        ):
            settings["mirror"] = False

        # ----------------------------------------------------
        # Create job
        # ----------------------------------------------------

        with video_jobs_lock:

            video_jobs[
                job_id
            ] = {

                "status": "queued",

                "progress": 0,

                "message": (
                    "Video queued."
                ),

                "error": None,

                "result_path": None,

                "filename": None,

                "size": 0,

                "created_at": time.time(),

                "settings": {
                    "effect": settings[
                        "effect"
                    ],
                    "line_size": settings[
                        "line_size"
                    ],
                    "box_size": settings[
                        "box_size"
                    ],
                    "text_size": settings[
                        "text_size"
                    ],
                    "mirror": settings[
                        "mirror"
                    ],
                    "face_color": settings[
                        "face_color"
                    ],
                    "hand_color": settings[
                        "hand_color"
                    ],
                    "object_color": settings[
                        "object_color"
                    ],
                    "text_color": settings[
                        "text_color"
                    ],
                },
            }

        # ----------------------------------------------------
        # Background processing thread
        # ----------------------------------------------------

        worker = threading.Thread(
            target=process_full_video_job,
            args=(
                job_id,
                input_path,
                job_dir,
                settings
            ),
            daemon=True
        )

        worker.start()

        return jsonify({

            "success": True,

            "job_id": job_id,

            "status": "queued",

            "message": (
                "Video berhasil di-upload "
                "dan masuk processing queue."
            )
        })

    except Exception as exc:

        cleanup_file(
            input_path
        )

        cleanup_directory(
            job_dir
        )

        print(
            "START FULL VIDEO ERROR:",
            exc
        )

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


# ============================================================
# FULL VIDEO STATUS
# ============================================================

@app.route(
    "/process_full_video_status/<job_id>",
    methods=["GET"]
)
def process_full_video_status(
    job_id
):

    with video_jobs_lock:

        job = video_jobs.get(
            job_id
        )

        if job is None:

            return jsonify({
                "success": False,
                "error": (
                    "Job ID tidak ditemukan."
                )
            }), 404

        return jsonify({

            "success": True,

            "job_id": job_id,

            "status": job.get(
                "status",
                "unknown"
            ),

            "progress": job.get(
                "progress",
                0
            ),

            "message": job.get(
                "message",
                ""
            ),

            "error": job.get(
                "error"
            ),

            "filename": job.get(
                "filename"
            ),

            "size": job.get(
                "size",
                0
            ),

            "width": job.get(
                "width"
            ),

            "height": job.get(
                "height"
            ),

            "rotation": job.get(
                "rotation",
                0
            ),

            "download_url": (
                f"/download_processed_video/"
                f"{job_id}"
                if job.get(
                    "status"
                ) == "completed"
                else None
            )
        })


# ============================================================
# DOWNLOAD FULL VIDEO
# ============================================================

@app.route(
    "/download_processed_video/<job_id>",
    methods=["GET"]
)
def download_processed_video(
    job_id
):

    with video_jobs_lock:

        job = video_jobs.get(
            job_id
        )

        if job is None:

            return jsonify({
                "success": False,
                "error": (
                    "Job ID tidak ditemukan."
                )
            }), 404

        status = job.get(
            "status"
        )

        if status != "completed":

            return jsonify({
                "success": False,
                "error": (
                    "Video belum selesai diproses."
                ),
                "status": status,
                "progress": job.get(
                    "progress",
                    0
                )
            }), 409

        result_path = job.get(
            "result_path"
        )

        filename = job.get(
            "filename",
            "blob_vision_processed.mp4"
        )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not result_path:

        return jsonify({
            "success": False,
            "error": (
                "Path hasil video tidak tersedia."
            )
        }), 404

    if not os.path.exists(
        result_path
    ):

        return jsonify({
            "success": False,
            "error": (
                "File hasil sudah tidak tersedia."
            )
        }), 404

    return send_file(
        result_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=filename
    )


# ============================================================
# DELETE OLD JOBS
# ============================================================

def cleanup_old_video_jobs(
    max_age_seconds=3600
):
    """
    Remove completed/error jobs older than one hour.
    """

    now = time.time()

    remove_ids = []

    with video_jobs_lock:

        for job_id, job in (
            list(
                video_jobs.items()
            )
        ):

            created_at = job.get(
                "created_at",
                now
            )

            age = (
                now - created_at
            )

            if age > max_age_seconds:

                remove_ids.append(
                    job_id
                )

    for job_id in remove_ids:

        with video_jobs_lock:

            job = video_jobs.pop(
                job_id,
                None
            )

        if not job:
            continue

        result_path = job.get(
            "result_path"
        )

        cleanup_file(
            result_path
        )


# ============================================================
# SHUTDOWN ENDPOINT
# ============================================================

@app.route(
    "/shutdown",
    methods=["POST"]
)
def shutdown():

    """
    Important:

    This endpoint intentionally does NOT call
    engine.close().

    A browser can trigger requests during page unload.
    Closing the global engine from there could kill
    the engine while Gunicorn is still running.
    """

    return jsonify({
        "success": True,
        "message": (
            "Server shutdown is managed "
            "by the hosting process."
        )
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(
    413
)
def request_too_large(error):

    return jsonify({
        "success": False,
        "error": (
            "File terlalu besar. "
            f"Maximum upload adalah "
            f"{MAX_UPLOAD_MB:.0f} MB."
        )
    }), 413


@app.errorhandler(
    404
)
def not_found(error):

    return jsonify({
        "success": False,
        "error": "Route tidak ditemukan."
    }), 404


@app.errorhandler(
    500
)
def internal_server_error(error):

    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# ============================================================
# CLEANUP
# ============================================================

@atexit.register
def cleanup():

    print(
        "Cleaning up BLOB VISION..."
    )

    try:

        with video_jobs_lock:

            jobs = list(
                video_jobs.values()
            )

            video_jobs.clear()

        for job in jobs:

            cleanup_file(
                job.get(
                    "result_path"
                )
            )

    except Exception as exc:

        print(
            "Job cleanup error:",
            exc
        )

    try:

        engine.close()

    except Exception as exc:

        print(
            "Engine cleanup error:",
            exc
        )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    # Render provides PORT automatically.
    # Local development falls back to 5001.

    try:

        port = int(
            os.environ.get(
                "PORT",
                "5001"
            )
        )

    except (
        TypeError,
        ValueError
    ):

        port = 5001

    host = os.environ.get(
        "HOST",
        "0.0.0.0"
    )

    print("========================================")
    print("BLOB VISION SERVER")
    print("========================================")
    print(
        "HOST:",
        host
    )
    print(
        "PORT:",
        port
    )
    print(
        "DEBUG:",
        False
    )
    print("========================================")

    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )
    