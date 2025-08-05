FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

RUN pip install fastapi uvicorn transformers accelerate \
    safetensors sentencepiece Pillow python-multipart firebase-admin && \
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

VOLUME /model_cache

ENV TRANSFORMERS_CACHE=/model_cache \
    HF_HOME=/model_cache

WORKDIR /app
COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
