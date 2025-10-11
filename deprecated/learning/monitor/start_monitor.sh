#!/bin/bash

set -e

echo "🔍 Starting VoiceStand Performance Monitor"
echo "========================================="

# Start Grafana in the background
echo "Starting Grafana..."
/run.sh &
GRAFANA_PID=$!

# Wait for Grafana to be ready
echo "Waiting for Grafana to start..."
until curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; do
    echo "Waiting for Grafana..."
    sleep 5
done

echo "✅ Grafana started successfully"

# Start Python monitoring services
echo "Starting performance monitor..."
cd /app

# Start performance monitor in background
python3 performance_monitor.py &
MONITOR_PID=$!

# Start metrics collector
python3 metrics_collector.py &
COLLECTOR_PID=$!

echo "✅ Monitoring services started"
echo "📊 Grafana Dashboard: http://localhost:3000"
echo "📈 Prometheus Metrics: http://localhost:8000"

# Function to handle shutdown
cleanup() {
    echo "🛑 Shutting down monitoring services..."
    kill $MONITOR_PID 2>/dev/null || true
    kill $COLLECTOR_PID 2>/dev/null || true
    kill $GRAFANA_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Wait for any process to exit
wait $GRAFANA_PID $MONITOR_PID $COLLECTOR_PID