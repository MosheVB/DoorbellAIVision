FROM nvcr.io/nvidia/tensorrt:25.09-py3

# GStreamer (with nvh264dec NVDEC hardware decoder) + OpenCV with GStreamer support.
# ffmpeg: repair incomplete MP4s (e.g. moov atom missing after abrupt stop).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ffmpeg \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir flask requests tzdata nvidia-ml-py psutil \
    "ultralytics>=8.3.0"

WORKDIR /app
