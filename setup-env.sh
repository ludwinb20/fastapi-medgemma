#!/bin/bash

echo "🔧 Configurando variables de entorno..."

# Crear archivo .env si no existe
if [ ! -f .env ]; then
    echo "📝 Creando archivo .env..."
    cat > .env << EOF
# Token de Hugging Face (requerido)
HF_TOKEN=tu_token_de_huggingface_aqui

# Configuración de Firebase (opcional)
FIREBASE_SERVICE_ACCOUNT_PATH=serviceAccountKey.json

# Configuración de cache (usar HF_HOME en lugar de TRANSFORMERS_CACHE)
HF_HOME=/model_cache
EOF
    echo "✅ Archivo .env creado"
    echo "⚠️  IMPORTANTE: Edita el archivo .env y configura tu HF_TOKEN real"
else
    echo "✅ Archivo .env ya existe"
fi

# Crear directorio de cache
mkdir -p model_cache

echo "🎉 Configuración completada!"
echo "📝 Recuerda editar .env con tu token real de Hugging Face" 
