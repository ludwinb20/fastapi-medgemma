from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
import os
from typing import Optional
from pydantic import BaseModel
from enum import Enum
from io import BytesIO
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedGemma API", version="1.0")

# Enum para tipos de estudio médico
class MedicalExamType(str, Enum):
    CHEST_XRAY = "radiografía de tórax"
    BRAIN_MRI = "resonancia magnética cerebral"
    ABDOMINAL_CT = "tomografía abdominal"

# Modelo de respuesta
class AnalysisResult(BaseModel):
    result: str
    model: str = "medgemma-4b-it"
    processing_time: Optional[float] = None

# Cargar modelo y processor
try:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN no está configurado")

    logger.info("Cargando modelo MedGemma...")
    processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        device_map="auto",
        torch_dtype=torch.float32,  # puedes usar bfloat16 si lo soporta tu hardware
        token=HF_TOKEN
    )
    print(model.hf_device_map)
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise RuntimeError(f"No se pudo cargar el modelo: {str(e)}")

# Función auxiliar para analizar imagen
def run_analysis(image: Image.Image, prompt: str) -> AnalysisResult:
    # Asegurarse de que la imagen sea RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Iniciar temporizador
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    # Mensajes estructurados
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    # Formatear prompt
    formatted_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Preparar inputs
    inputs = processor(
        text=formatted_prompt,
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to("cuda")

    # Generar salida
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Decodificar y limpiar
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    result = result.replace(formatted_prompt, "").strip()

    end_time.record()
    torch.cuda.synchronize()
    processing_time = start_time.elapsed_time(end_time) / 1000

    return AnalysisResult(
        result=result,
        processing_time=processing_time
    )

# Endpoint 1: análisis rápido con prompt fijo
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_medical_image(
    file: UploadFile = File(...),
    prompt: str = "Describe los hallazgos médicos relevantes en español"
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        image = Image.open(BytesIO(await file.read()))
        return run_analysis(image, prompt)

    except UnidentifiedImageError:
        raise HTTPException(400, "Formato de imagen no soportado")
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(500, "Memoria GPU insuficiente")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error interno: {str(e)}")

# Endpoint 2: análisis con tipo de examen y notas clínicas
@app.post("/analyze-async", response_model=AnalysisResult)
async def analyze_with_context(
    file: UploadFile = File(...),
    exam_type: MedicalExamType = MedicalExamType.CHEST_XRAY,
    clinical_notes: Optional[str] = ""
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Solo se aceptan imágenes")
        
        image = Image.open(BytesIO(await file.read()))

        prompt = f"Analiza esta {exam_type.value}. Antecedentes clínicos: {clinical_notes}"
        return run_analysis(image, prompt)

    except UnidentifiedImageError:
        raise HTTPException(400, "Formato de imagen no soportado")
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(500, "Memoria GPU insuficiente")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error interno: {str(e)}")
