from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import atexit

from tracking_engine import TrackingEngine


app = Flask(__name__)

print("Starting Tracking Engine...")
engine = TrackingEngine()
print("Tracking Engine loaded.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.get_data()

    if not data:
        return Response("No frame received", status=400)

    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        return Response("Invalid frame", status=400)

    effect = request.headers.get("X-Tracking-Effect", "medium").lower()

    if effect not in ("low", "medium", "high"):
        effect = "medium"

    try:
        line = float(request.headers.get("X-Tracking-Line", "100")) / 100.0
        box = float(request.headers.get("X-Tracking-Box", "100")) / 100.0
        text_size = float(request.headers.get("X-Tracking-Text", "100")) / 100.0
        marker = float(request.headers.get("X-Tracking-Marker", "100")) / 100.0

        processed = engine.process(
            frame,
            mirror=True,
            effect=effect,
            line=line,
            box=box,
            text_size=text_size,
            marker=marker
        )

        if processed is None:
            return Response("Processing failed", status=500)

        ok, encoded = cv2.imencode(
            ".jpg",
            processed,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if not ok:
            return Response("JPEG encoding failed", status=500)

        return Response(
            encoded.tobytes(),
            mimetype="image/jpeg"
        )

    except Exception as e:
        print("PROCESS FRAME ERROR:", e)
        return Response("Processing error", status=500)


@app.route("/stats")
def stats():
    return jsonify(engine.get_stats())


@app.route("/reset_tracking", methods=["POST"])
def reset_tracking():
    try:
        engine.reset()
        return jsonify({"success": True})
    except Exception as e:
        print("RESET ERROR:", e)
        return jsonify({"success": False}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        engine.close()
    except Exception:
        pass

    return jsonify({"success": True})


@atexit.register
def cleanup():
    try:
        engine.close()
    except Exception:
        pass


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
