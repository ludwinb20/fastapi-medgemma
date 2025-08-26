import os
import json
import logging
from typing import Generator
from transformers import TextIteratorStreamer
from threading import Thread
import torch

# Configuración de logging
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Eres LucasMed, un asistente médico de IA especializado en apoyar a médicos en la práctica clínica.\n"
    "\n"
    "⚠️ REGLA PRINCIPAL, ABSOLUTA Y PRIORITARIA ⚠️\n"
    "SOLO puedes responder a preguntas relacionadas con medicina y salud. "
    "Si el tema NO es médico, debes responder ÚNICAMENTE con la frase fija: "
    "'Soy un asistente médico especializado. Solo puedo responder a consultas médicas.'\n"
    "No intentes dar contexto adicional, explicaciones ni disculpas. "
    "No reformules esta frase. No inventes otra variante. "
    "Esta regla es PRIORITARIA sobre cualquier otra.\n"
    "\n"
    "SOLO PUEDES responder a preguntas relacionadas con:\n"
    "✅ Temas permitidos:\n"
    "- Diagnósticos médicos\n"
    "- Tratamientos médicos\n"
    "- Fisiología y anatomía\n"
    "- Farmacología\n"
    "- Análisis clínicos\n"
    "- Síntomas y enfermedades\n"
    "- Procedimientos médicos\n"
    "\n"
    "❌ Temas prohibidos:\n"
    "- Política, economía, deportes\n"
    "- Historia, geografía, entretenimiento\n"
    "- Cualquier tema NO médico\n"
    "\n"
    "- Responde SIEMPRE en español.\n"
    "- Utiliza lenguaje profesional y técnico, adecuado para médicos.\n"
    "- Sé claro, preciso y completo, aportando información relevante para la toma de decisiones clínicas.\n"
    "- Estructura tus respuestas médicas de manera lógica: "
    "1) Resumen breve del punto clave, "
    "2) Explicación clínica detallada, "
    "3) Diagnósticos diferenciales o consideraciones adicionales, "
    "4) Recomendaciones basadas en evidencia o guías clínicas cuando sea posible.\n"
    "- Incluye información adicional relevante: fisiopatología, protocolos de manejo, tratamientos de primera línea, "
    "indicaciones y contraindicaciones, prevención y pronóstico.\n"
    "- Responde solo al último mensaje del usuario, usando contexto si es necesario, pero no repitas respuestas previas.\n"
    "- Aclara cuando la evidencia no sea concluyente o se requiera criterio clínico individualizado.\n"
    "- Señala cuando sea necesario confirmar en guías locales, consensos clínicos o protocolos hospitalarios.\n"
)



def get_system_prompt() -> str:
    """Obtiene el prompt del sistema desde variables de entorno o usa el default"""
    return os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

def get_medical_image_prompt() -> str:
    return (
        "Eres LucasMed, un experto radiólogo y analista de imágenes médicas, especializado en apoyar a médicos en la "
        "interpretación de estudios diagnósticos.\n"
        "\n"
            "⚠️ REGLA PRINCIPAL, ABSOLUTA Y PRIORITARIA ⚠️\n"
        "SOLO puedes responder a preguntas relacionadas con medicina y salud. "
        "Si el tema NO es médico, debes responder ÚNICAMENTE con la frase fija: "
        "'Soy un asistente médico especializado. Solo puedo responder a consultas médicas.'\n"
        "No intentes dar contexto adicional, explicaciones ni disculpas. "
        "No reformules esta frase. No inventes otra variante. "
        "Esta regla es PRIORITARIA sobre cualquier otra.\n"
        "SOLO PUEDES responder a preguntas relacionadas con:\n"
        "- Medicina y salud\n"
        "- Diagnósticos médicos\n"
        "- Tratamientos médicos\n"
        "- Fisiología y anatomía\n"
        "- Farmacología\n"
        "- Análisis clínicos\n"
        "- Síntomas y enfermedades\n"
        "- Procedimientos médicos\n"
        "- Interpretación de imágenes médicas\n"
        "\n"
        "❌ NO PUEDES responder a:\n"
        "- Política, economía, deportes\n"
        "- Historia, geografía, entretenimiento\n"
        "- Cualquier tema NO médico\n"
        "\n"
        "Si te preguntan algo NO médico, responde ÚNICAMENTE:\n"
        "'Soy un asistente médico especializado. Solo puedo ayudarte con consultas relacionadas con medicina y salud. Por favor, consulta con un especialista en el tema que necesitas.'\n"
        "\n"
        "- Responde SIEMPRE en español.\n"
        "- Utiliza un lenguaje técnico y preciso, adecuado para profesionales de la salud.\n"
        "- Estructura tus análisis en el siguiente orden: "
        "1) Descripción general de la imagen, "
        "2) Hallazgos específicos relevantes, "
        "3) Diagnósticos diferenciales posibles, "
        "4) Recomendaciones (estudios adicionales, correlación clínica o manejo sugerido según guías).\n"
        "- Destaca hallazgos radiológicos de importancia clínica y su posible correlación con la historia del paciente.\n"
        "- Si la imagen es ambigua, indica las limitaciones diagnósticas y qué estudios adicionales podrían aclarar los hallazgos.\n"
        "- Si la imagen no es médica o no puede interpretarse adecuadamente, indícalo claramente.\n"
        "- Siempre que sea posible, menciona guías clínicas relevantes o criterios radiológicos reconocidos "
        "(ej. BI-RADS, Fleischner Society, ACR, etc.).\n"
    )

def get_exam_report_prompt() -> str:
    return (
        "Eres LucasMed, un radiólogo experto especializado en análisis de imágenes médicas. "
        "Tu función es analizar imágenes médicas de diferentes tipos y generar reportes completos y profesionales.\n"
        "\n"
        "⚠️ REGLA PRINCIPAL, ABSOLUTA Y PRIORITARIA ⚠️\n"
        "SOLO puedes analizar imágenes médicas y generar reportes médicos. "
        "Si la imagen NO es médica o no puede interpretarse, debes responder ÚNICAMENTE con la frase fija: "
        "'Esta imagen no corresponde a un estudio médico válido o no puede ser interpretada adecuadamente.'\n"
        "No intentes dar contexto adicional, explicaciones ni disculpas. "
        "No reformules esta frase. No inventes otra variante. "
        "Esta regla es PRIORITARIA sobre cualquier otra.\n"
        "\n"
        "INSTRUCCIONES ESPECÍFICAS PARA REPORTES DE EXAMEN:\n"
        "- Analiza la imagen médica del tipo especificado con precisión técnica.\n"
        "- Proporciona un análisis detallado en lenguaje médico técnico y profesional.\n"
        "- Identifica todos los hallazgos relevantes, tanto normales como anormales.\n"
        "- Utiliza terminología médica estándar y criterios radiológicos reconocidos.\n"
        "- Estructura tu respuesta en formato JSON con las claves: summary, findings, disclaimer.\n"
        "\n"
        "FORMATO DE RESPUESTA REQUERIDO:\n"
        "Debes responder ÚNICAMENTE en formato JSON válido con la siguiente estructura:\n"
        "{\n"
        '  "summary": "Análisis detallado en lenguaje médico técnico y profesional",\n'
        '  "findings": "Lista clara y detallada de todos los hallazgos potenciales",\n'
        '  "disclaimer": "Importante: Este es un análisis preliminar generado por IA y no debe considerarse un diagnóstico médico definitivo. La interpretación de imágenes médicas es compleja y debe ser realizada por un radiólogo certificado. Consulte a un profesional de la salud para una evaluación completa y un diagnóstico preciso."\n'
        "}\n"
        "\n"
        "REGLAS ADICIONALES:\n"
        "- Responde SIEMPRE en español.\n"
        "- Utiliza lenguaje médico técnico y profesional.\n"
        "- Sé preciso, completo y objetivo en tu análisis.\n"
        "- Incluye hallazgos normales y anormales relevantes.\n"
        "- Menciona limitaciones diagnósticas si las hay.\n"
        "- Sugiere estudios adicionales cuando sea apropiado.\n"
        "- NO incluyas texto adicional fuera del JSON.\n"
        "- NO agregues comentarios, explicaciones ni texto fuera del formato JSON especificado.\n"
    )

def clean_response(full_response, prompt):
    """Limpia la respuesta removiendo el prompt original y repeticiones"""
    # Buscar el último token de asistente en el prompt
    assistant_markers = ["<|im_start|>assistant", "<|im_end|>", "<|im_start|>user", "<|im_end|>"]
    
    # Intentar diferentes estrategias de limpieza
    cleaned = full_response
    
    # Estrategia 1: Buscar después del último marcador de asistente
    for marker in assistant_markers:
        if marker in prompt:
            parts = prompt.split(marker)
            if len(parts) > 1:
                prompt_until_assistant = parts[0] + marker
                if prompt_until_assistant in full_response:
                    cleaned = full_response.split(prompt_until_assistant)[-1]
                    break
    
    # Estrategia 2: Si no funciona, buscar después de "assistant"
    if cleaned == full_response and "assistant" in prompt.lower():
        assistant_index = prompt.lower().find("assistant")
        if assistant_index != -1:
            prompt_until_assistant = prompt[:assistant_index] + "assistant"
            if prompt_until_assistant in full_response:
                cleaned = full_response.split(prompt_until_assistant)[-1]
    
    # Estrategia 3: Buscar después de "model" (común en algunos modelos)
    if cleaned == full_response and "model" in full_response.lower():
        model_index = full_response.lower().find("model")
        if model_index != -1:
            cleaned = full_response[model_index + 5:]  # "model" tiene 5 caracteres
    
    # Estrategia 4: Buscar después de "Responde al último mensaje"
    if cleaned == full_response and "Responde al último mensaje" in prompt:
        respond_index = full_response.find("Responde al último mensaje")
        if respond_index != -1:
            # Buscar después de esta frase en la respuesta
            prompt_after_respond = prompt[respond_index:]
            if prompt_after_respond in full_response:
                cleaned = full_response.split(prompt_after_respond)[-1]
    
    # Estrategia 5: Remover el prompt completo si está al inicio
    if cleaned == full_response and prompt in full_response:
        cleaned = full_response.replace(prompt, "")
    
    # Estrategia 6: Buscar el inicio real de la respuesta (después del prompt)
    if cleaned == full_response:
        # Buscar patrones comunes que indican el inicio de la respuesta
        response_patterns = [
            "Hola, lamento escuchar",
            "Entiendo que",
            "Para ayudarte mejor",
            "¿Podrías decirme",
            "Te recomiendo",
            "Mientras tanto",
            "¡Hola!",
            "Hola,"
        ]
        
        for pattern in response_patterns:
            if pattern in full_response:
                pattern_index = full_response.find(pattern)
                if pattern_index > len(prompt) * 0.8:  # Si está después del 80% del prompt
                    cleaned = full_response[pattern_index:]
                    break
    
    # Limpiar marcadores de chat si quedan
    for marker in assistant_markers:
        cleaned = cleaned.replace(marker, "")
    
    # Limpiar líneas vacías al inicio
    cleaned = cleaned.lstrip()
    
    # Remover repeticiones excesivas
    sentences = cleaned.split('.')
    if len(sentences) > 2:
        # Mantener solo oraciones únicas
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        # Si hay muchas repeticiones, mantener solo las primeras 2-3 oraciones únicas
        if len(unique_sentences) > 3:
            unique_sentences = unique_sentences[:3]
        
        cleaned = '. '.join(unique_sentences) + ('.' if cleaned.endswith('.') else '')
    
    return cleaned.strip()

def is_trivial_question(text: str) -> bool:
    text = text.strip()
    # Preguntas triviales: cortas, directas, sin contexto ni razonamiento
    if len(text) < 30 and text.endswith("?"):
        return True
    # Preguntas tipo saludo o confirmación
    if text.lower() in {"hola", "gracias", "ok", "buenos días"}:
        return True
    return False

def generate_stream_response(model, processor, formatted_prompt, user_input=None, max_new_tokens=1100):
    """Genera respuesta en streaming real usando TextIteratorStreamer"""
    logger.info(f"Iniciando generación con max_new_tokens={max_new_tokens}")
    
    # Procesar con el modelo
    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt"
    ).to("cuda")
    
    logger.info(f"Prompt procesado, tokens de entrada: {len(inputs.input_ids[0])}")

    # Streamer que produce texto incrementalmente
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Parámetros optimizados para respuestas más extensas y detalladas
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,           # Habilitar muestreo para mayor diversidad
        temperature=0.7,          # Temperatura moderada para balance entre creatividad y coherencia
        top_p=0.9,               # Nucleus sampling para mejor calidad
        repetition_penalty=1.1,   # Penalizar repeticiones
        length_penalty=1.0,       # No penalizar respuestas largas
        early_stopping=False,     # Permitir que la respuesta se complete naturalmente
        streamer=streamer,
    )

    # Ejecutar la generación en un hilo separado
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    assistant_markers = ["<|im_start|>assistant", "<|im_end|>", "<|im_start|>user", "<|im_end|>"]
    full_response = ""

    try:
        for new_text in streamer:
            # Limpiar marcadores de chat
            for marker in assistant_markers:
                new_text = new_text.replace(marker, "")
            
            if new_text:
                full_response += new_text
                
                # Detectar repeticiones (menos agresivo)
                sentences = full_response.split('.')
                if len(sentences) > 5:
                    # Verificar si las últimas 5 oraciones son exactamente iguales
                    recent_sentences = sentences[-5:]
                    if len(set(recent_sentences)) == 1 and len(recent_sentences[0].strip()) > 20:
                        # Detener solo si hay repetición excesiva y clara
                        logger.info("Detectada repetición excesiva, deteniendo generación")
                        break
                
                yield f"data: {json.dumps({'token': new_text, 'finished': False})}\n\n"
    finally:
        thread.join()
        logger.info(f"Generación completada. Respuesta total: {len(full_response)} caracteres")

    # Señalizar fin
    yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"

def generate_stream_response_with_images(model, processor, formatted_prompt, images, user_input=None, max_new_tokens=1100):
    """Genera respuesta en streaming real usando TextIteratorStreamer para contenido multimodal"""
    logger.info(f"Iniciando generación multimodal con max_new_tokens={max_new_tokens}")
    
    # Validar que hay imágenes para procesar
    if not images or len(images) == 0:
        logger.warning("No hay imágenes para procesar, usando generación de texto normal")
        # Si no hay imágenes, usar la función de texto normal
        return generate_stream_response(model, processor, formatted_prompt, user_input, max_new_tokens)
    
    # Procesar con el modelo incluyendo imágenes
    logger.info(f"Procesando {len(images)} imágenes con el modelo")
    inputs = processor(
        text=formatted_prompt,
        images=images,
        return_tensors="pt"
    ).to("cuda")
    
    logger.info(f"Prompt multimodal procesado, tokens de entrada: {len(inputs.input_ids[0])}")
    logger.info(f"Tipo de inputs: {type(inputs)}")
    logger.info(f"Claves de inputs: {inputs.keys() if hasattr(inputs, 'keys') else 'No es un dict'}")

    # Streamer que produce texto incrementalmente
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Parámetros optimizados para respuestas más extensas y detalladas
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,           # Habilitar muestreo para mayor diversidad
        temperature=0.7,          # Temperatura moderada para balance entre creatividad y coherencia
        top_p=0.9,               # Nucleus sampling para mejor calidad
        repetition_penalty=1.1,   # Penalizar repeticiones
        length_penalty=1.0,       # No penalizar respuestas largas
        early_stopping=False,     # Permitir que la respuesta se complete naturalmente
        streamer=streamer,
    )

    # Ejecutar la generación en un hilo separado
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    assistant_markers = ["<|im_start|>assistant", "<|im_end|>", "<|im_start|>user", "<|im_end|>"]
    full_response = ""

    try:
        token_count = 0
        for new_text in streamer:
            # Limpiar marcadores de chat
            for marker in assistant_markers:
                new_text = new_text.replace(marker, "")
            
            if new_text:
                full_response += new_text
                token_count += 1
                
                # Detectar repeticiones (menos agresivo)
                sentences = full_response.split('.')
                if len(sentences) > 5:
                    # Verificar si las últimas 5 oraciones son exactamente iguales
                    recent_sentences = sentences[-5:]
                    if len(set(recent_sentences)) == 1 and len(recent_sentences[0].strip()) > 20:
                        # Detener solo si hay repetición excesiva y clara
                        logger.info("Detectada repetición excesiva, deteniendo generación")
                        break
                
                yield f"data: {json.dumps({'token': new_text, 'finished': False})}\n\n"
        
        # Si no se generó ningún token, enviar un mensaje de error
        if token_count == 0:
            logger.warning("No se generaron tokens, enviando mensaje de error")
            yield f"data: {json.dumps({'token': 'No se pudo generar una respuesta. Por favor, intenta de nuevo.', 'finished': False})}\n\n"
            
    except Exception as e:
        logger.error(f"Error durante la generación multimodal: {str(e)}")
        yield f"data: {json.dumps({'token': f'Error en la generación: {str(e)}', 'finished': False})}\n\n"
    finally:
        thread.join()
        logger.info(f"Generación multimodal completada. Tokens generados: {token_count}, Respuesta total: {len(full_response)} caracteres")

    # Señalizar fin
    yield f"data: {json.dumps({'token': '', 'finished': True})}\n\n"

def clean_context_from_streaming_errors(context: str) -> str:
    """Limpia el contexto de errores de streaming y datos JSON"""
    if not context:
        return context
    
    # Remover líneas que contengan errores de streaming
    lines = context.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Ignorar líneas que contengan errores de streaming
        if any(error_pattern in line.lower() for error_pattern in [
            '{"error":', 'http 500', 'internal server error', 
            'finished:', 'token:', '{"token"'
        ]):
            continue
            
        # Ignorar líneas que empiecen con 'data:' pero NO sean imágenes
        if line.startswith('data:') and not line.startswith('data:image/'):
            continue
            
        # Ignorar líneas que sean solo JSON
        if line.startswith('{') and line.endswith('}'):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def process_context_messages(context: str) -> list:
    """Procesa el contexto y construye la lista de mensajes dinámicamente"""
    messages = []
    
    # Limpiar el contexto de posibles errores de streaming
    context = clean_context_from_streaming_errors(context)
    
    # Procesar el contexto línea por línea
    context_lines = context.strip().split('\n')
    current_message = None
    current_role = None
    
    for line in context_lines:
        line = line.strip()
        if not line:
            continue
            
        # Detectar si es una imagen (data URI)
        is_image = line.startswith('data:image/')
        
        # Detectar si es inicio de un nuevo mensaje
        # Formato 1: [Usuario] o [Asistente]
        if line.startswith('[Usuario]') or line.startswith('[User]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('[Asistente]') or line.startswith('[Assistant]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        # Formato 2: user: o assistant: (formato original)
        elif line.startswith('user:') or line.startswith('User:') or line.startswith('Usuario:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('assistant:') or line.startswith('Assistant:') or line.startswith('Asistente:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        elif is_image:
            # Es una imagen, convertir data URI a PIL Image
            try:
                # Extraer la parte base64
                header, encoded = line.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_bytes))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Agregar la imagen al mensaje actual o crear uno nuevo
                if current_message and current_role == "user":
                    # Agregar la imagen al mensaje de usuario actual
                    current_message["content"].append({"type": "image", "image": image})
                else:
                    # Crear nuevo mensaje de usuario con la imagen
                    if current_message:
                        messages.append(current_message)
                    
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image}
                        ]
                    }
                    current_role = "user"
                    
            except Exception as e:
                logger.error(f"Error procesando imagen en contexto: {str(e)}")
                # Si hay error, tratar como texto
                if current_message:
                    text_content = None
                    for content in current_message["content"]:
                        if content["type"] == "text":
                            text_content = content
                            break
                    
                    if text_content:
                        text_content["text"] = f"{text_content['text']}\n{line}"
                    else:
                        current_message["content"].append({"type": "text", "text": line})
                else:
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": line}
                        ]
                    }
                    current_role = "user"
            
        else:
            # Si no tiene prefijo, es continuación del mensaje actual
            if current_message:
                # Buscar si ya hay contenido de texto
                text_content = None
                for content in current_message["content"]:
                    if content["type"] == "text":
                        text_content = content
                        break
                
                if text_content:
                    # Agregar la línea al texto existente
                    text_content["text"] = f"{text_content['text']}\n{line}"
                else:
                    # Agregar nuevo contenido de texto
                    current_message["content"].append({"type": "text", "text": line})
            else:
                # Si no hay mensaje actual, asumir que es mensaje de usuario
                current_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": line}
                    ]
                }
                current_role = "user"
    
    # Agregar el último mensaje si existe
    if current_message:
        messages.append(current_message)
    
    # Asegurar que los roles alternen correctamente
    cleaned_messages = []
    last_role = None
    
    for message in messages:
        current_role = message["role"]
        
        # Si es el primer mensaje, agregarlo
        if last_role is None:
            cleaned_messages.append(message)
            last_role = current_role
            continue
        
        # Si el rol actual es diferente al anterior, agregarlo
        if current_role != last_role:
            cleaned_messages.append(message)
            last_role = current_role
        else:
            # Si hay roles consecutivos iguales, combinar el contenido
            if cleaned_messages:
                # Combinar el contenido del mensaje actual con el último mensaje del mismo rol
                last_message = cleaned_messages[-1]
                
                # Agregar todo el contenido del mensaje actual al último mensaje
                for content in message["content"]:
                    if content["type"] == "text":
                        # Buscar si ya existe contenido de texto
                        existing_text = None
                        for existing_content in last_message["content"]:
                            if existing_content["type"] == "text":
                                existing_text = existing_content
                                break
                        
                        if existing_text:
                            existing_text["text"] = f"{existing_text['text']}\n{content['text']}"
                        else:
                            last_message["content"].append(content)
                    else:
                        # Para imágenes, agregar directamente
                        last_message["content"].append(content)
    
    print("*****************************************cleaned_messages****************************************")
    print(cleaned_messages)
    
    # Contar imágenes en los mensajes
    image_count = 0
    for message in cleaned_messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_count += 1
    
    logger.info(f"Procesamiento completado. {len(cleaned_messages)} mensajes, {image_count} imágenes encontradas")
    
    return cleaned_messages

def process_context_messages_with_images(context: str) -> list:
    """Procesa el contexto y construye la lista de mensajes dinámicamente, incluyendo imágenes"""
    from PIL import Image
    import base64
    from io import BytesIO
    
    messages = []
    
    # Limpiar el contexto de posibles errores de streaming
    context = clean_context_from_streaming_errors(context)
    
    # Procesar el contexto línea por línea
    context_lines = context.strip().split('\n')
    current_message = None
    current_role = None
    
    for line in context_lines:
        line = line.strip()
        if not line:
            continue
            
        # Detectar si es una imagen (data URI)
        is_image = line.startswith('data:image/')
        
        # Detectar si es inicio de un nuevo mensaje
        # Formato 1: [Usuario] o [Asistente]
        if line.startswith('[Usuario]') or line.startswith('[User]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('[Asistente]') or line.startswith('[Assistant]'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(']', 1)[1].strip() if ']' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        # Formato 2: user: o assistant: (formato original)
        elif line.startswith('user:') or line.startswith('User:') or line.startswith('Usuario:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de usuario
            user_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
            current_role = "user"
            
        elif line.startswith('assistant:') or line.startswith('Assistant:') or line.startswith('Asistente:'):
            # Guardar mensaje anterior si existe
            if current_message:
                messages.append(current_message)
            
            # Iniciar nuevo mensaje de asistente
            assistant_message = line.split(':', 1)[1].strip() if ':' in line else line
            current_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
            current_role = "assistant"
            
        elif is_image:
            # Es una imagen, convertir data URI a PIL Image
            try:
                # Extraer la parte base64
                header, encoded = line.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_bytes))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Agregar la imagen al mensaje actual o crear uno nuevo
                if current_message and current_role == "user":
                    # Agregar la imagen al mensaje de usuario actual
                    current_message["content"].append({"type": "image", "image": image})
                else:
                    # Crear nuevo mensaje de usuario con la imagen
                    if current_message:
                        messages.append(current_message)
                    
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image}
                        ]
                    }
                    current_role = "user"
                    
            except Exception as e:
                logger.error(f"Error procesando imagen en contexto: {str(e)}")
                # Si hay error, tratar como texto
                if current_message:
                    text_content = None
                    for content in current_message["content"]:
                        if content["type"] == "text":
                            text_content = content
                            break
                    
                    if text_content:
                        text_content["text"] = f"{text_content['text']}\n{line}"
                    else:
                        current_message["content"].append({"type": "text", "text": line})
                else:
                    current_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": line}
                        ]
                    }
                    current_role = "user"
            
        else:
            # Si no tiene prefijo, es continuación del mensaje actual
            if current_message:
                # Buscar si ya hay contenido de texto
                text_content = None
                for content in current_message["content"]:
                    if content["type"] == "text":
                        text_content = content
                        break
                
                if text_content:
                    # Agregar la línea al texto existente
                    text_content["text"] = f"{text_content['text']}\n{line}"
                else:
                    # Agregar nuevo contenido de texto
                    current_message["content"].append({"type": "text", "text": line})
            else:
                # Si no hay mensaje actual, asumir que es mensaje de usuario
                current_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": line}
                    ]
                }
                current_role = "user"
    
    # Agregar el último mensaje si existe
    if current_message:
        messages.append(current_message)
    
    # Asegurar que los roles alternen correctamente
    cleaned_messages = []
    last_role = None
    
    for message in messages:
        current_role = message["role"]
        
        # Si es el primer mensaje, agregarlo
        if last_role is None:
            cleaned_messages.append(message)
            last_role = current_role
            continue
        
        # Si el rol actual es diferente al anterior, agregarlo
        if current_role != last_role:
            cleaned_messages.append(message)
            last_role = current_role
        else:
            # Si hay roles consecutivos iguales, combinar el contenido
            if cleaned_messages:
                # Combinar el contenido del mensaje actual con el último mensaje del mismo rol
                last_message = cleaned_messages[-1]
                
                # Agregar todo el contenido del mensaje actual al último mensaje
                for content in message["content"]:
                    if content["type"] == "text":
                        # Buscar si ya existe contenido de texto
                        existing_text = None
                        for existing_content in last_message["content"]:
                            if existing_content["type"] == "text":
                                existing_text = existing_content
                                break
                        
                        if existing_text:
                            existing_text["text"] = f"{existing_text['text']}\n{content['text']}"
                        else:
                            last_message["content"].append(content)
                    else:
                        # Para imágenes, agregar directamente
                        last_message["content"].append(content)
    
    print("*****************************************cleaned_messages_with_images****************************************")
    print(cleaned_messages)
    
    return cleaned_messages