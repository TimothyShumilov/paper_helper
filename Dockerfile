FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install llama-cpp-python without CUDA for portability;
# for GPU support set CMAKE_ARGS="-DGGML_CUDA=ON" in build args
ARG CMAKE_ARGS="-DGGML_CUDA=OFF"
RUN CMAKE_ARGS="${CMAKE_ARGS}" pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/pdfs /app/data/models

CMD ["python", "-m", "main"]
