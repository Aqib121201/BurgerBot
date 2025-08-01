#!/bin/bash

# BürgerBot Docker Entrypoint Script

set -e

echo "🇩🇪 Starting BürgerBot..."

# Check if we should run the pipeline first
if [ "$RUN_PIPELINE" = "true" ]; then
    echo "Running data preprocessing and model training pipeline..."
    python run_pipeline.py --full
fi

# Check if we should run the app
if [ "$RUN_APP" = "true" ] || [ -z "$RUN_APP" ]; then
    echo "Starting Streamlit application..."
    exec streamlit run app/app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
else
    echo "Container started successfully. Use 'docker exec' to run commands."
    exec "$@"
fi 