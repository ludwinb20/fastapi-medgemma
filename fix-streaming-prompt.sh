#!/bin/bash

echo "🔧 Corrigiendo problema del prompt en streaming..."

# Detener contenedor actual
if docker ps | grep -q medgemma-api; then
    echo "🛑 Deteniendo contenedor actual..."
    docker stop medgemma-api
    docker rm medgemma-api
fi

# Reconstruir con corrección
echo "🔨 Reconstruyendo imagen con corrección..."
docker build -t medgemma-api .

# Verificar que se construyó correctamente
if docker images | grep -q medgemma-api; then
    echo "✅ Imagen reconstruida correctamente"
else
    echo "❌ Error al reconstruir la imagen"
    exit 1
fi

# Ejecutar nuevo contenedor
echo "🐳 Ejecutando nuevo contenedor..."
docker run -d \
  --name medgemma-api \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/model_cache:/model_cache \
  --restart unless-stopped \
  medgemma-api

echo "✅ Corrección aplicada!"
echo "📊 Verificando estado..."

# Esperar a que inicie
sleep 15

# Verificar logs
echo "📋 Últimos logs:"
docker logs --tail 20 medgemma-api

echo ""
echo "🎉 ¡Corrección aplicada!"
echo "🧪 Para probar el streaming corregido:"
echo "   python3 test-streaming.py" 