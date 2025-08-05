#!/usr/bin/env python3
"""
Script para probar el streaming de la API MedGemma
"""

import requests
import json
import time

def test_text_streaming():
    """Prueba el streaming de texto"""
    print("🧪 Probando streaming de texto...")
    
    url = "http://localhost:8000/api/process-text-stream"
    
    # Datos de prueba
    data = {
        "prompt": "Explica qué es la diabetes de manera simple",
        "context": "Eres un asistente médico que explica conceptos médicos de forma clara"
    }
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming iniciado correctamente")
            print("📝 Respuesta en tiempo real:")
            print("-" * 50)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remover 'data: '
                            if 'token' in data:
                                print(data['token'], end='', flush=True)
                            if data.get('finished'):
                                print("\n✅ Streaming completado")
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

def test_image_streaming():
    """Prueba el streaming de imagen"""
    print("\n🧪 Probando streaming de imagen...")
    
    url = "http://localhost:8000/api/process-image-stream"
    
    # Crear una imagen de prueba simple (1x1 pixel)
    import base64
    from PIL import Image
    import io
    
    # Crear imagen de prueba
    img = Image.new('RGB', (100, 100), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    
    # Datos de prueba
    data = {
        "imageDataUri": f"data:image/png;base64,{img_str}",
        "prompt": "Describe esta imagen"
    }
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming de imagen iniciado correctamente")
            print("📝 Respuesta en tiempo real:")
            print("-" * 50)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remover 'data: '
                            if 'token' in data:
                                print(data['token'], end='', flush=True)
                            if data.get('finished'):
                                print("\n✅ Streaming de imagen completado")
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de streaming...")
    print("=" * 50)
    
    # Probar streaming de texto
    test_text_streaming()
    
    # Probar streaming de imagen
    test_image_streaming()
    
    print("\n🎉 Pruebas completadas!") 