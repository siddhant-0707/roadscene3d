#!/bin/bash
# Start MLflow tracking server with UI

MLFLOW_TRACKING_URI=${1:-./mlruns}
MLFLOW_PORT=${2:-5000}

echo "🚀 Starting MLflow Server"
echo "📊 Tracking URI: $MLFLOW_TRACKING_URI"
echo "🌐 Port: $MLFLOW_PORT"
echo ""
echo "Access the UI at: http://localhost:$MLFLOW_PORT"
echo ""

mlflow ui \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --port "$MLFLOW_PORT" \
    --host 0.0.0.0

