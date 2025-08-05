from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
import os
from typing import Optional
from pydantic import BaseModel
import logging
from io import BytesIO
import json
import firebase_admin
from firebase_admin import auth, credentials

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Image Analysis API", version="1.0")

# Inicializar Firebase Admin
try:
    # Puedes usar una variable de entorno para la ruta del archivo
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar Firebase Admin: {str(e)}")
    # Continuar sin Firebase si no está configurado
    pass

async def verify_api_key(request: Request):
    """
    Middleware para verificar API Key en lugar de session cookies.
    """
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(status_code=401, detail="No API key provided")
    
    # Obtener la API key desde variables de entorno
    expected_api_key = os.getenv("MEDGEMMA_API_KEY")
    
    if not expected_api_key:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    
    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Si la API key es válida, retornar información básica
    return {
        "authenticated": True,
        "auth_method": "api_key",
        "timestamp": "2024-01-01T00:00:00Z"  # Puedes agregar timestamp real si quieres
   }


async def verify_session_cookie(request: Request):
    session_cookie = request.cookies.get("__session")
    print(1)
    print(request.cookies)

    if not session_cookie:
        raise HTTPException(status_code=401, detail="No session cookie")
    
    try:
        # Verificar la session cookie
        print(2)
        decoded_claims = auth.verify_session_cookie(session_cookie, check_revoked=True)
        print(decoded_claims)
        return decoded_claims  # Contiene uid, email, etc.
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid session")

# Modelo de respuesta
class AnalysisResult(BaseModel):
    result: str
    model: str = "medgemma-4b-it"
    processing_time: Optional[float] = None

# Nuevos modelos para los endpoints
class TextProcessRequest(BaseModel):
    prompt: str
    context: Optional[str] = None

class ImageProcessRequest(BaseModel):
    imageDataUri: str
    prompt: str

class ProcessResponse(BaseModel):
    response: str
    tokens_used: int
    success: bool

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
    prompt: str = "Describe los hallazgos médicos relevantes en español",
    user_claims: dict = Depends(verify_api_key)
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

@app.post("/api/process-text", response_model=ProcessResponse)
async def process_text(
    request: TextProcessRequest,
    user_claims: dict = Depends(verify_api_key)
):
    """Procesa texto usando MedGemma-4b-it"""
    try:
        logger.info(f"Procesando texto para usuario: {user_claims.get('uid', 'unknown')}")
        
        # Construir prompt con contexto si está disponible
        full_prompt = request.prompt
        if request.context:
            full_prompt = f"Contexto: {request.context}\n\nPregunta: {request.prompt}"
        
        # Estructura de mensajes para texto
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt}
                ]
            }
        ]

        # Aplicar template de chat
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Procesar con el modelo
        inputs = processor(
            text=formatted_prompt,
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
        
        # Contar tokens (aproximado)
        tokens_used = len(inputs.input_ids[0]) + len(outputs[0]) - len(inputs.input_ids[0])
        
        logger.info(f"Procesamiento de texto completado. Tokens usados: {tokens_used}")
        
        return ProcessResponse(
            response=result,
            tokens_used=tokens_used,
            success=True
        )

    except Exception as e:
        logger.error(f"Error procesando texto: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/api/process-image", response_model=ProcessResponse)
async def process_image(
    request: ImageProcessRequest,
    user_claims: dict = Depends(verify_api_key)
):
    """Procesa imagen usando MedGemma-4b-it"""
    try:
        logger.info(f"Procesando imagen para usuario: {user_claims.get('uid', 'unknown')}")
        
        # Validar y decodificar data URI
        if not request.imageDataUri.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Formato de imagen inválido")
        
        try:
            # Extraer la parte base64
            header, encoded = request.imageDataUri.split(",", 1)
            image_data = BytesIO(encoded.encode('utf-8'))
            
            # Decodificar base64
            import base64
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail="Error decodificando imagen")

        # Estructura de mensajes para imagen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        # Aplicar template de chat
        formatted_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Procesar con el modelo
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
        
        # Contar tokens (aproximado)
        tokens_used = len(inputs.input_ids[0]) + len(outputs[0]) - len(inputs.input_ids[0])
        
        logger.info(f"Procesamiento de imagen completado. Tokens usados: {tokens_used}")
        
        return ProcessResponse(
            response=result,
            tokens_used=tokens_used,
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
