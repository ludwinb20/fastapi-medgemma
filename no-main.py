from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
import os
from typing import Optional
from pydantic import BaseModel
import logging
from io import BytesIO
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Image Analysis API", version="1.0")

# Modelo de respuesta
class AnalysisResult(BaseModel):
    result: str
    model: str = "medgemma-4b-it"
    processing_time: Optional[float] = None

# Cargar el modelo (con manejo de errores)
try:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN no está configurado")
    
    logger.info("Cargando modelo MedGemma...")
    processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        device_map="auto",
        torch_dtype=torch.float32,
        token=HF_TOKEN
    )
    logger.info("Modelo cargado correctamente")

except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise RuntimeError(f"No se pudo cargar el modelo: {str(e)}")

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_medical_image(
    file: UploadFile = File(..., description="Imagen médica para análisis"),
    prompt: str = "Describe los hallazgos médicos relevantes en español"
):
    """Analiza imágenes médicas usando MedGemma-4b-it con formato estructurado"""
    try:
        logger.info(f"Procesando archivo: {file.filename}")
        
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        # Procesamiento con temporización
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Cargar y verificar imagen
        try:
            image_data = await file.read()
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

        # Estructura de mensajes similar al ejemplo de Gemma-3
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        # Aplicar template de chat (adaptado para MedGemma)
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Procesar con el modelo (similar al collate_fn del ejemplo)
        inputs = processor(
            text=formatted_prompt,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        # Generar respuesta
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Decodificar y limpiar respuesta
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        result = result.replace(formatted_prompt, "").strip()
        
        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time) / 1000

        logger.info(f"Análisis completado en {processing_time:.2f}s")
        
        return {
            "result": result,
            "processing_time": processing_time
        }

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("Error: Memoria GPU insuficiente")
        raise HTTPException(status_code=500, detail="Memoria GPU insuficiente para procesar la imagen")
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")