FROM python:3.12-slim

WORKDIR /miner

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GPU Backend selection — uncomment the one you need:
# ── NVIDIA CUDA ──────────────────────────────────────────────────────────────
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip install transformers accelerate bitsandbytes
#
# ── AMD/Intel Vulkan ─────────────────────────────────────────────────────────
# RUN CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python
#
# ── Apple Silicon (MLX) ──────────────────────────────────────────────────────
# Not supported in Docker on non-Apple hosts.
# Use the MLX backend natively on macOS.

COPY . .

ENTRYPOINT ["python", "miner.py"]
