# Development container for proposal-policy-vo with GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # Development tools
    git \
    wget \
    curl \
    vim \
    build-essential \
    # For debugging
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Install Python development tools
RUN pip install --no-cache-dir \
    pylint \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    ipython \
    ipdb

# Copy and install project dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Install matplotlib for visualization
RUN pip install --no-cache-dir matplotlib

# Install GPU-accelerated libraries
# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install GPU-accelerated OpenCV (optional, uncomment if needed)
# RUN pip uninstall -y opencv-python && \
#     pip install --no-cache-dir opencv-contrib-python

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["/bin/bash"]
