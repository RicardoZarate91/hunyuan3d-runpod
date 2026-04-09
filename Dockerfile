FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
LABEL org.opencontainers.image.source=https://github.com/RicardoZarate91/hunyuan3d-runpod

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libegl-mesa0 \
    ninja-build \
    blender \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# ── Clone Hunyuan3D-Omni (shape generation with controls) ──
# No CUDA extensions needed — pure Python/PyTorch
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git /app/omni

WORKDIR /app

# Install Python dependencies (matches Hunyuan3D-Omni requirements.txt)
RUN pip install --no-cache-dir \
    ninja pybind11 \
    diffusers==0.30.0 einops opencv-python-headless \
    numpy==1.24.4 scipy==1.14.1 transformers==4.46.0 \
    omegaconf pyyaml tqdm trimesh pymeshlab==2023.12.post3 \
    pygltflib xatlas accelerate==1.1.1 \
    pytorch-lightning==1.9.5 torchdiffeq==0.2.5 timm==1.0.20 \
    rembg onnxruntime \
    safetensors huggingface_hub hf_transfer \
    imageio pillow psutil \
    open3d==0.19.0 cupy-cuda12x==13.4.1

# Install torchaudio (CUDA 12.4, same index as torch)
RUN pip install --no-cache-dir \
    torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Model weights are downloaded on first boot (too large for GH Actions disk).
# On RunPod, use a network volume mounted at /workspace to cache models.
# The handler downloads to /workspace/weights/ on cold start if not present.

# Copy our handler + Roblox pipeline
COPY handler.py /app/handler.py
COPY retopo.py /app/retopo.py
COPY blender_decimate.py /app/blender_decimate.py
COPY blender_postprocess.py /app/blender_postprocess.py
COPY blender_accessory.py /app/blender_accessory.py
COPY postprocess_clothing.py /app/postprocess_clothing.py

# Copy Roblox templates
COPY roblox-templates/ /opt/roblox-templates/

# Headless rendering
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Make Omni package importable
ENV PYTHONPATH="/app/omni"

CMD ["python3", "/app/handler.py"]
