#!/bin/bash

echo "🔧 Solucionando problema de PyTorch..."

# Detener contenedor actual
if docker ps | grep -q medgemma-api; then
    echo "🛑 Deteniendo contenedor actual..."
    docker stop medgemma-api
    docker rm medgemma-api
fi

# Limpiar imagen anterior
echo "🧹 Limpiando imagen anterior..."
docker rmi medgemma-api 2>/dev/null || true

# Reconstruir con correcciones
echo "🔨 Reconstruyendo imagen con correcciones..."
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

echo "✅ Reconstrucción completada!"
echo "📊 Verificando estado..."

# Esperar a que inicie
sleep 15

# Verificar logs
echo "📋 Últimos logs:"
docker logs --tail 20 medgemma-api

echo ""
echo "🎉 ¡Reconstrucción completada!"
echo "📖 Verifica el estado con: docker logs -f medgemma-api" 