FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp/blobvision-home YOLO_CONFIG_DIR=/tmp/blobvision-home/Ultralytics MPLCONFIGDIR=/tmp/blobvision-home/matplotlib MAX_UPLOAD_MB=1024

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libgomp1 && rm -rf /var/lib/apt/lists/* && mkdir -p /tmp/blobvision-home/Ultralytics /tmp/blobvision-home/matplotlib

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0 && pip install -r requirements.txt

COPY app.py .
COPY tracking_engine.py .
COPY templates ./templates
COPY models ./models
COPY yolo11n.pt .

RUN test -f app.py && test -f tracking_engine.py && test -f templates/index.html && test -f models/face_detection_yunet_2023mar.onnx && test -f models/emotion-ferplus-8.onnx && test -f yolo11n.pt

EXPOSE 10000

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 0 --graceful-timeout 30 --access-logfile - --error-logfile - app:app"]
