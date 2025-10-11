#!/bin/bash

# VoiceStand Learning System Startup Script
# This script starts the complete learning infrastructure

set -e

echo "🚀 Starting VoiceStand Learning System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p training_data learning_data optimization_logs fine_tuned_models
mkdir -p models/ensemble models/fine_tuned models/uk_specialized

# Start the learning services
echo "🐳 Starting Docker services..."
docker-compose -f docker-compose.learning.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check database
if docker-compose -f docker-compose.learning.yml exec -T voicestand-learning-db pg_isready -U voicestand; then
    echo "✅ Database is ready"
else
    echo "❌ Database is not ready"
fi

# Check learning API
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Learning API is ready"
else
    echo "⏳ Learning API is starting up..."
fi

# Check monitoring
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Monitoring is ready"
else
    echo "⏳ Monitoring is starting up..."
fi

echo ""
echo "🎉 VoiceStand Learning System is starting up!"
echo ""
echo "📊 Services:"
echo "  - Learning API: http://localhost:8080"
echo "  - Health Check: http://localhost:8080/health"
echo "  - API Docs: http://localhost:8080/docs"
echo "  - Monitoring: http://localhost:3000"
echo "  - Database: localhost:5433"
echo ""
echo "📋 To view logs:"
echo "  docker-compose -f docker-compose.learning.yml logs -f"
echo ""
echo "🛑 To stop:"
echo "  docker-compose -f docker-compose.learning.yml down"
echo ""