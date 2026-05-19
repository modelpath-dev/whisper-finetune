FROM python:3.10-slim

WORKDIR /app

# System libraries needed for audio I/O (soundfile -> libsndfile, librosa -> ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU build of PyTorch first so the heavy CUDA wheel isn't pulled.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default to running the test suite; override for training/eval.
CMD ["pytest"]
