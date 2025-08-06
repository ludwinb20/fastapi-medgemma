#!/usr/bin/env python3
"""
Script para probar la limpieza de prompts específicos
"""

import requests
import json

def test_prompt_cleaning():
    """Prueba la limpieza de prompts específicos"""
    print("🧪 Probando limpieza de prompts...")
    
    url = "http://localhost:8000/api/process-text"
    
    # Datos de prueba (el caso problemático)
    data = {
        "prompt": "Eres LucasMed, un asistente médico de IA en un chat con un doctor. Usa el contexto de los últimos mensajes para dar una respuesta precisa y útil.\n\nHistorial de mensajes:\nai: Lo siento, no pude procesar tu solicitud. Intenta nuevamente más tarde.\nuser: hola\nuser: me duele la cabeza\n\nResponde al último mensaje del usuario de la forma más útil y profesional posible. Si hay imágenes, tenlas en cuenta en tu análisis.",
        "context": "ai: Lo siento, no pude procesar tu solicitud. Intenta nuevamente más tarde.\nuser: hola\nuser: me duele la cabeza"
    }
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Respuesta recibida")
            print("📝 Respuesta limpia:")
            print("-" * 50)
            print(result.get('response', 'Sin respuesta'))
            print("-" * 50)
            
            # Verificar si el prompt original está en la respuesta
            original_prompt = data["prompt"]
            response_text = result.get('response', '')
            
            if original_prompt in response_text:
                print("❌ PROBLEMA: El prompt original está en la respuesta")
                print("🔍 Buscando el prompt en la respuesta...")
                prompt_index = response_text.find(original_prompt)
                if prompt_index != -1:
                    print(f"📍 Prompt encontrado en posición: {prompt_index}")
                    print("📄 Respuesta completa:")
                    print(response_text)
            else:
                print("✅ ÉXITO: El prompt no está en la respuesta")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

def test_simple_prompt():
    """Prueba con un prompt simple"""
    print("\n🧪 Probando prompt simple...")
    
    url = "http://localhost:8000/api/process-text"
    
    data = {
        "prompt": "Hola, ¿cómo estás?",
        "context": "Eres un asistente amigable"
    }
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Respuesta recibida")
            print("📝 Respuesta:")
            print(result.get('response', 'Sin respuesta'))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de limpieza de prompts...")
    print("=" * 50)
    
    # Probar el caso problemático
    test_prompt_cleaning()
    
    # Probar caso simple
    test_simple_prompt()
    
    print("\n🎉 Pruebas completadas!") 