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
    "Eres LucasMed, un asistente médico de IA.\n"
    "- Responde SIEMPRE en español.\n"
    "- Sé claro, profesional y conciso.\n"
    "- Responde solo al último mensaje del usuario usando el contexto si existe.\n"
    "- No uses el formato 'input:'/'output:'.\n"
    "- Incluye advertencias de seguridad solo cuando sea relevante.\n"
    "- Como parte del contexto, vas a recibir mensajes enviados por el usuario y mensajes enviados por el asistente. No repitas respuestas del asistente, ni redundes en ellas.\n"
    "- Responde solamente a la pregunta sin agregar advertencias o introducciones innecesarias, a menos que el usuario las solicite.\n"
)

def get_system_prompt() -> str:
    """Obtiene el prompt del sistema desde variables de entorno o usa el default"""
    return os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

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

def generate_stream_response(model, processor, formatted_prompt, user_input=None, max_new_tokens=500):
    """Genera respuesta en streaming real usando TextIteratorStreamer"""
    # Procesar con el modelo
    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt"
    ).to("cuda")

    # Streamer que produce texto incrementalmente
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Parámetros simples como en el ejemplo oficial
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
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
                
                # Detectar repeticiones
                sentences = full_response.split('.')
                if len(sentences) > 3:
                    # Verificar si las últimas 3 oraciones son similares
                    recent_sentences = sentences[-3:]
                    if len(set(recent_sentences)) == 1 and len(recent_sentences[0].strip()) > 10:
                        # Detener si hay repetición excesiva
                        break
                
                yield f"data: {json.dumps({'token': new_text, 'finished': False})}\n\n"
    finally:
        thread.join()

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
            'data:', '{"error":', 'http 500', 'internal server error', 
            'finished:', 'token:', '{"token"'
        ]):
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
            
        else:
            # Si no tiene prefijo, es continuación del mensaje actual
            if current_message:
                # Agregar la línea al mensaje actual
                current_text = current_message["content"][0]["text"]
                current_message["content"][0]["text"] = f"{current_text}\n{line}"
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
                # Combinar el texto del mensaje actual con el último mensaje del mismo rol
                current_text = message["content"][0]["text"]
                last_text = cleaned_messages[-1]["content"][0]["text"]
                combined_text = f"{last_text}\n{current_text}"
                cleaned_messages[-1]["content"][0]["text"] = combined_text
    
    print("*****************************************cleaned_messages****************************************")
    print(cleaned_messages)
    
    return cleaned_messages