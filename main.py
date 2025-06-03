from fastapi import FastAPI, UploadFile, File
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os

app = FastAPI()

# Cargar el modelo
token = os.environ["HF_TOKEN"]
processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=token)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it", device_map="auto", torch_dtype=torch.float16
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image = Image.open(file.file)
    inputs = processor(images=image, text="Describe los hallazgos médicos", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    return {"result": result}