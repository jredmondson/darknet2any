# Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1) System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-dev python3-venv python3-pip \
      build-essential ca-certificates git libssl-dev libffi-dev \
      libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 2) pip tools + venv
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# 3) Only copy in requirements and install them
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Default command (you can override this on `docker run` if you like)
CMD ["python", "--version"]
