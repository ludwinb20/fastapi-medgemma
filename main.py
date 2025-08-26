from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
import os
from typing import Optional, Generator
from pydantic import BaseModel
import logging
from io import BytesIO
import json
import firebase_admin
from firebase_admin import auth, credentials
from utils import get_system_prompt, get_medical_image_prompt, get_exam_report_prompt, clean_response, clean_json_response, ExamReportOutputParser, generate_stream_response, generate_stream_response_with_images, process_context_messages, process_context_messages_with_images

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

# Modelos de respuesta

# Nuevos modelos para los endpoints
class TextProcessRequest(BaseModel):
    prompt: str
    context: Optional[str] = None

class ImageProcessRequest(BaseModel):
    imageDataUri: str
    prompt: str
    context: Optional[str] = None

class ExamReportRequest(BaseModel):
    imageDataUri: str
    examType: str


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



@app.post("/api/process-text", response_model=ProcessResponse)
async def process_text(
    request: TextProcessRequest,
    user_claims: dict = Depends(verify_api_key)
):
    """Procesa texto usando MedGemma-4b-it"""
    try:
        logger.info(f"Procesando texto para usuario: {user_claims.get('uid', 'unknown')}")
        
        # Construir mensajes con contexto si está disponible
        if request.context:
            # Verificar si hay imágenes reales en el contexto (data URI válidos)
            has_images = "data:image/" in request.context and any(
                line.strip().startswith("data:image/") and "," in line 
                for line in request.context.split('\n')
            )
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_medical_image_prompt() if has_images else get_system_prompt()}
                    ]
                }
            ]
            
            if has_images:
                # Si hay imágenes, usar la función que maneja imágenes
                context_messages = process_context_messages_with_images(request.context)
                messages.extend(context_messages)
                
                # Recopilar todas las imágenes del contexto
                all_images = []
                for message in context_messages:
                    if message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "image":
                                all_images.append(content["image"])
                
                # Agregar el mensaje actual del usuario
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt}
                    ]
                })
                
                # Aplicar template de chat
                formatted_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Procesar con el modelo incluyendo imágenes
                inputs = processor(
                    text=formatted_prompt,
                    images=all_images,
                    return_tensors="pt"
                ).to("cuda")
            else:
                # Si no hay imágenes, usar la función de texto normal
                context_messages = process_context_messages(request.context)
                messages.extend(context_messages)
                
                # Agregar el mensaje actual del usuario
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt}
                    ]
                })
                
                # Aplicar template de chat
                formatted_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Procesar con el modelo
                inputs = processor(
                    text=formatted_prompt,
                    return_tensors="pt"
                ).to("cuda")
        else:
            # Sin contexto, usar estructura simple
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_system_prompt()}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt}
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
                return_tensors="pt"
            ).to("cuda")

        # Generar respuesta con parámetros optimizados para respuestas más extensas
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Aumentado de 800 a 2048 para respuestas más largas
            do_sample=True,       # Habilitar muestreo para mayor diversidad
            temperature=0.7,      # Temperatura moderada para balance entre creatividad y coherencia
            top_p=0.9,           # Nucleus sampling para mejor calidad
            repetition_penalty=1.1,  # Penalizar repeticiones
            length_penalty=1.0,   # No penalizar respuestas largas
            early_stopping=False  # Permitir que la respuesta se complete naturalmente
        )

        # Decodificar y limpiar respuesta
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        result = clean_response(result, formatted_prompt)
        
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

@app.post("/api/process-text-stream")
async def process_text_stream(
    request: TextProcessRequest,
    user_claims: dict = Depends(verify_api_key)
):
    """Procesa texto usando MedGemma-4b-it con streaming"""
    try:
        logger.info(f"Procesando texto en streaming para usuario: {user_claims.get('uid', 'unknown')}")
        print(f"DEBUG: Contexto recibido: {request.context[:200] if request.context else 'None'}...")
        print(f"DEBUG: Longitud del contexto: {len(request.context) if request.context else 0}")
        print(f"DEBUG: Contexto completo: {request.context}")
        
        # Construir mensajes con contexto si está disponible
        if request.context:
            # Verificar si hay imágenes reales en el contexto (data URI válidos)
            has_images = "data:image/" in request.context and any(
                line.strip().startswith("data:image/") and "," in line 
                for line in request.context.split('\n')
            )
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_medical_image_prompt() if has_images else get_system_prompt()}
                    ]
                }
            ]
            
            if has_images:
                # Si hay imágenes, usar la función que maneja imágenes
                context_messages = process_context_messages_with_images(request.context)
                messages.extend(context_messages)
                
                # Recopilar todas las imágenes del contexto
                all_images = []
                for message in context_messages:
                    if message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "image":
                                all_images.append(content["image"])
                
                # Agregar el mensaje actual del usuario
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt}
                    ]
                })
                
                # Aplicar template de chat
                formatted_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                def generate_stream():
                    try:
                        for chunk in generate_stream_response_with_images(model, processor, formatted_prompt, all_images, request.prompt, max_new_tokens=1500):
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error en streaming con imágenes: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"
            else:
                # Si no hay imágenes, usar la función de texto normal
                context_messages = process_context_messages(request.context)
                messages.extend(context_messages)
                
                # Agregar el mensaje actual del usuario
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt}
                    ]
                })
                
                # Aplicar template de chat
                formatted_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                def generate_stream():
                    try:
                        for chunk in generate_stream_response(model, processor, formatted_prompt, request.prompt, max_new_tokens=1500):
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error en streaming: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"
        else:
            # Sin contexto, usar estructura simple
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_system_prompt()}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt}
                    ]
                }
            ]
            
            print("*****************************************messages****************************************")
            print(messages)

            # Aplicar template de chat
            formatted_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            def generate_stream():
                try:
                    for chunk in generate_stream_response(model, processor, formatted_prompt, request.prompt, max_new_tokens=1500):
                        yield chunk
                except Exception as e:
                    logger.error(f"Error in streaming: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except Exception as e:
        logger.error(f"Error procesando texto en streaming: {str(e)}", exc_info=True)
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

        # Construir mensajes con contexto si está disponible
        if request.context:
            # Si hay contexto, procesar los mensajes dinámicamente incluyendo imágenes
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_medical_image_prompt()}
                    ]
                }
            ]
            
            # Procesar el contexto usando la función que maneja imágenes
            context_messages = process_context_messages_with_images(request.context)
            messages.extend(context_messages)
            
            # Agregar el mensaje actual del usuario con la imagen
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "image", "image": image}
                ]
            })
        else:
            # Sin contexto, usar estructura simple
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_system_prompt()}
                    ]
                },
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

        # Recopilar todas las imágenes del contexto y la imagen actual
        all_images = [image]
        
        # Si hay contexto, extraer imágenes del contexto
        if request.context:
            context_messages = process_context_messages_with_images(request.context)
            for message in context_messages:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "image":
                            all_images.append(content["image"])

        # Procesar con el modelo
        inputs = processor(
            text=formatted_prompt,
            images=all_images,
            return_tensors="pt"
        ).to("cuda")

        # Generar respuesta con parámetros optimizados para respuestas más extensas
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Aumentado de 800 a 2048 para respuestas más largas
            do_sample=True,       # Habilitar muestreo para mayor diversidad
            temperature=0.7,      # Temperatura moderada para balance entre creatividad y coherencia
            top_p=0.9,           # Nucleus sampling para mejor calidad
            repetition_penalty=1.1,  # Penalizar repeticiones
            length_penalty=1.0,   # No penalizar respuestas largas
            early_stopping=False  # Permitir que la respuesta se complete naturalmente
        )

        # Decodificar y limpiar respuesta
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        result = clean_response(result, formatted_prompt)
        
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

@app.post("/api/process-image-stream")
async def process_image_stream(
    request: ImageProcessRequest,
    user_claims: dict = Depends(verify_api_key)
):
    """Procesa imagen usando MedGemma-4b-it con streaming"""
    try:
        logger.info(f"Procesando imagen en streaming para usuario: {user_claims.get('uid', 'unknown')}")
        
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

        # Construir mensajes con contexto si está disponible
        if request.context:
            # Si hay contexto, procesar los mensajes dinámicamente incluyendo imágenes
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_medical_image_prompt()}
                    ]
                }
            ]
            
            # Procesar el contexto usando la función que maneja imágenes
            context_messages = process_context_messages_with_images(request.context)
            messages.extend(context_messages)
            
            # Agregar el mensaje actual del usuario con la imagen
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "image", "image": image}
                ]
            })
        else:
            # Sin contexto, usar estructura simple
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": get_medical_image_prompt()}
                    ]
                },
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

        def generate_stream():
            try:
                # Recopilar todas las imágenes del contexto y la imagen actual
                all_images = [image]
                logger.info(f"Imagen actual agregada, total: {len(all_images)}")
                
                # Si hay contexto, extraer imágenes del contexto
                if request.context:
                    context_messages = process_context_messages_with_images(request.context)
                    logger.info(f"Contexto procesado, {len(context_messages)} mensajes encontrados")
                    
                    for i, message in enumerate(context_messages):
                        if message["role"] == "user":
                            for content in message["content"]:
                                if content["type"] == "image":
                                    all_images.append(content["image"])
                                    logger.info(f"Imagen {len(all_images)} agregada desde mensaje {i}")
                
                logger.info(f"Total de imágenes a procesar: {len(all_images)}")
                
                for chunk in generate_stream_response_with_images(model, processor, formatted_prompt, all_images, request.prompt, max_new_tokens=1500):
                    yield chunk
            except Exception as e:
                logger.error(f"Error en streaming de imagen: {str(e)}")
                yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando imagen en streaming: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/api/exam-report", response_model=ProcessResponse)
async def generate_exam_report(
    request: ExamReportRequest,
    user_claims: dict = Depends(verify_api_key)
):
    """Genera un reporte completo de examen médico basado en una imagen"""
    try:
        logger.info(f"Generando reporte de examen para usuario: {user_claims.get('uid', 'unknown')}")
        
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

        # Construir el prompt específico para el tipo de examen
        exam_prompt = f"Analiza esta imagen médica de tipo '{request.examType}' y proporciona un reporte completo."
        
        # Construir mensajes con el prompt del sistema para reportes de examen
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": get_exam_report_prompt()}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": exam_prompt},
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
            return_tensors="pt"
        ).to("cuda")

        # Generar respuesta con parámetros optimizados para reportes detallados
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Aumentado para reportes más detallados
            do_sample=True,       # Habilitar muestreo para mayor diversidad
            temperature=0.7,      # Temperatura moderada para balance entre creatividad y coherencia
            top_p=0.9,           # Nucleus sampling para mejor calidad
            repetition_penalty=1.1,  # Penalizar repeticiones
            length_penalty=1.0,   # No penalizar respuestas largas
            early_stopping=False  # Permitir que la respuesta se complete naturalmente
        )

        # Decodificar y limpiar respuesta
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        result = clean_response(result, formatted_prompt)
        
        # Contar tokens (aproximado)
        tokens_used = len(inputs.input_ids[0]) + len(outputs[0]) - len(inputs.input_ids[0])
        
        # Usar OutputParser para procesar la respuesta
        logger.info(f"Respuesta del modelo (longitud: {len(result)}): {result}...")
        
        # Verificar si la respuesta es muy larga
        if len(result) > 2000:
            logger.warning(f"Respuesta muy larga ({len(result)} chars), puede estar truncada")
        
        parser = ExamReportOutputParser()
        parsed_report = parser.parse(result)
        
        # Formatear para la API
        formatted_response = parser.format_for_api(parsed_report)
        
        # Determinar si fue exitoso basado en si se pudo extraer información válida
        success = all(key in parsed_report and parsed_report[key] for key in ['summary', 'findings'])
        
        logger.info(f"Reporte de examen procesado. Tokens usados: {tokens_used}, Éxito: {success}")
        
        return ProcessResponse(
            response=formatted_response,
            tokens_used=tokens_used,
            success=success
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando reporte de examen: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


