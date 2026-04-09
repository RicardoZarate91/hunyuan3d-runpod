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
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git /app/omni

# ── Clone Hunyuan3D-2.1 (texture/paint pipeline only) ──
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/paint

WORKDIR /app

# Install Python dependencies (merged from both repos)
RUN pip install --no-cache-dir \
    ninja pybind11 \
    diffusers==0.30.0 einops opencv-python-headless \
    numpy transformers==4.46.0 \
    omegaconf tqdm trimesh pymeshlab==2023.12.post3 \
    pygltflib xatlas accelerate==1.1.1 \
    rembg onnxruntime \
    safetensors huggingface_hub hf_transfer \
    imageio pillow psutil \
    open3d==0.19.0 cupy-cuda12x==13.4.1 \
    realesrgan pytorch-lightning==1.9.5 \
    bpy==4.0

# Build CUDA extensions for Paint pipeline (texture generation)
# Omni (shape) needs NO CUDA extensions
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
    && cd /app/paint/hy3dpaint/custom_rasterizer && pip install --no-build-isolation -e . \
    && cd /app/paint/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# ── Download model weights (baked into image for fast cold starts) ──
# Omni shape model (~13GB)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-Omni', local_dir='/app/weights/Hunyuan3D-Omni'); \
print('Omni shape model downloaded!')"

# Paint PBR texture model (~5GB for paint subfolder)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('tencent/Hunyuan3D-2.1', local_dir='/app/weights/Hunyuan3D-2.1'); \
print('Paint PBR model downloaded!')"

# Download RealESRGAN weights (needed by paint pipeline)
RUN mkdir -p /app/paint/hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -O /app/paint/hy3dpaint/ckpt/RealESRGAN_x4plus.pth

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

# Make both packages importable
ENV PYTHONPATH="/app/omni:/app/paint:${PYTHONPATH}"

CMD ["python3", "/app/handler.py"]
