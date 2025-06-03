FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

RUN pip install fastapi uvicorn transformers accelerate \
    safetensors sentencepiece Pillow

WORKDIR /app
COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
