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

# Clone Hunyuan3D-2.0
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    ninja pybind11 \
    diffusers einops opencv-python-headless \
    numpy transformers torchvision \
    omegaconf tqdm trimesh pymeshlab \
    pygltflib xatlas accelerate \
    rembg onnxruntime \
    safetensors huggingface_hub hf_transfer \
    imageio pillow psutil

# Build custom CUDA extensions (rasterizer + differentiable renderer)
RUN cd /app/hy3dgen/texgen/custom_rasterizer && pip install -e . \
    && cd /app/hy3dgen/texgen/differentiable_renderer && pip install -e .

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Pre-download models from HuggingFace (baked into image for fast cold starts)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-2', local_dir='/app/weights/Hunyuan3D-2'); \
print('Models downloaded!')"

# Copy our handler + Roblox pipeline
COPY handler.py /app/handler.py
COPY retopo.py /app/retopo.py
COPY blender_decimate.py /app/blender_decimate.py
COPY blender_postprocess.py /app/blender_postprocess.py
COPY blender_accessory.py /app/blender_accessory.py
COPY postprocess_clothing.py /app/postprocess_clothing.py

# Copy Roblox templates (reuse from trellis2-runpod)
COPY roblox-templates/ /opt/roblox-templates/

# Headless rendering
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

CMD ["python3", "/app/handler.py"]
