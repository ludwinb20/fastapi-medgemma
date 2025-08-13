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

def generate_stream_response(model, processor, formatted_prompt, user_input, max_new_tokens=500):
    """Genera respuesta en streaming real usando TextIteratorStreamer"""
    # Procesar con el modelo
    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to("cuda")

    # Streamer que produce texto incrementalmente
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    if is_trivial_question(user_input):
        do_sample = False
        temperature = 0.1
    else:
        do_sample = True
        temperature = 0.4
        
        
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        top_p=0.9,
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=1.15,  # Penalizar repeticiones
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
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

def process_context_messages(context: str) -> list:
    """Procesa el contexto y construye la lista de mensajes dinámicamente"""
    messages = []
    
    # Procesar el contexto línea por línea
    context_lines = context.strip().split('\n')
    for line in context_lines:
        line = line.strip()
        if not line:
            continue
            
        # Detectar si es mensaje de usuario o asistente
        if line.startswith('user:') or line.startswith('User:') or line.startswith('Usuario:'):
            # Mensaje de usuario
            user_message = line.split(':', 1)[1].strip() if ':' in line else line
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            })
        elif line.startswith('assistant:') or line.startswith('Assistant:') or line.startswith('Asistente:'):
            # Mensaje de asistente
            assistant_message = line.split(':', 1)[1].strip() if ':' in line else line
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            })
        else:
            # Si no tiene prefijo, asumir que es mensaje de usuario
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": line}
                ]
            })
    
    return messages

def is_trivial_question(text: str) -> bool:
    text = text.strip()
    # Preguntas triviales: cortas, directas, sin contexto ni razonamiento
    if len(text) < 30 and text.endswith("?"):
        return True
    # Preguntas tipo saludo o confirmación
    if text.lower() in {"hola", "gracias", "ok", "buenos días"}:
        return True
    return False