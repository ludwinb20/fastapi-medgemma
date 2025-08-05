#!/bin/bash

# Script de despliegue para MedGemma API
set -e

echo "🚀 Iniciando despliegue de MedGemma API..."

# Crear .env si no existe
if [ ! -f .env ]; then
    echo "📝 Creando archivo .env..."
    cat > .env << EOF
HF_TOKEN=tu_token_de_huggingface_aqui
FIREBASE_SERVICE_ACCOUNT_PATH=serviceAccountKey.json
HF_HOME=/model_cache
EOF
    echo "✅ Archivo .env creado"
    echo "⚠️  IMPORTANTE: Edita el archivo .env y configura tu HF_TOKEN real"
    echo "   nano .env"
    exit 1
else
    echo "✅ Archivo .env ya existe"
fi

# Verificar que HF_TOKEN esté configurado correctamente
source .env
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "tu_token_de_huggingface_aqui" ]; then
    echo "❌ Error: HF_TOKEN no está configurado correctamente"
    echo "Por favor, edita el archivo .env y configura tu token real:"
    echo "   nano .env"
    echo "   Cambia 'tu_token_de_huggingface_aqui' por tu token real"
    exit 1
fi

# Crear directorio de cache si no existe
mkdir -p model_cache

# Detener contenedor existente si existe
if docker ps -a | grep -q medgemma-api; then
    echo "🛑 Deteniendo contenedor existente..."
    docker stop medgemma-api || true
    docker rm medgemma-api || true
fi

# Construir imagen
echo "🔨 Construyendo imagen Docker..."
docker build -t medgemma-api .

# Ejecutar contenedor
echo "🐳 Ejecutando contenedor..."
docker run -d \
  --name medgemma-api \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/model_cache:/model_cache \
  --restart unless-stopped \
  medgemma-api

echo "✅ Despliegue completado!"
echo "📊 Verificando estado..."

# Esperar un poco para que el servicio inicie
sleep 10

# Verificar que esté funcionando
if curl -f http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ API está funcionando correctamente"
    echo "📖 Documentación disponible en: http://localhost:8000/docs"
else
    echo "⚠️  API puede estar aún iniciando..."
    echo "📋 Ver logs con: docker logs -f medgemma-api"
fi

echo "🎉 ¡Despliegue completado exitosamente!" 
