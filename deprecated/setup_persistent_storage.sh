#!/bin/bash

# VoiceStand Learning System - PostgreSQL Persistent Storage Setup
# This script sets up PostgreSQL persistent storage for the learning system

set -e  # Exit on any error

echo "🚀 Setting up VoiceStand PostgreSQL Persistent Storage..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Docker is installed and running
print_header "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Docker is installed and running ✅"

# Check if docker-compose is available
print_header "Checking Docker Compose availability..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    print_error "Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

print_status "Docker Compose is available ✅"

# Stop any existing containers
print_header "Stopping existing containers..."
$COMPOSE_CMD -f docker-compose.learning.yml down 2>/dev/null || true
print_status "Existing containers stopped"

# Create necessary directories
print_header "Creating necessary directories..."
mkdir -p models
mkdir -p learning_data
mkdir -p optimization_logs
mkdir -p training_data
mkdir -p fine_tuned_models
print_status "Directories created ✅"

# Start PostgreSQL with pgvector
print_header "Starting PostgreSQL database..."
$COMPOSE_CMD -f docker-compose.learning.yml up -d voicestand-learning-db

# Wait for database to be ready
print_status "Waiting for database to be ready..."
sleep 10

# Check if database is responding
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec $(docker ps -qf "name=voicestand-learning-db") pg_isready -U voicestand -d voicestand_learning &> /dev/null; then
        print_status "Database is ready ✅"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -eq $max_attempts ]; then
        print_error "Database failed to start after $max_attempts attempts"
        exit 1
    fi

    echo -n "."
    sleep 2
done

# Run database migrations
print_header "Running database migrations..."
cd learning/gateway

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r ../requirements.gateway.txt
else
    source venv/bin/activate
fi

# Run migrations
python migrate.py init

if [ $? -eq 0 ]; then
    print_status "Database migrations completed ✅"
else
    print_error "Database migrations failed ❌"
    exit 1
fi

# Verify the setup
print_header "Verifying database setup..."
python migrate.py verify

if [ $? -eq 0 ]; then
    print_status "Database verification passed ✅"
else
    print_error "Database verification failed ❌"
    exit 1
fi

# Show database status
print_header "Database status:"
python migrate.py status

cd ../..

# Update environment file
print_header "Creating environment configuration..."
cat > .env << EOF
# VoiceStand Learning System Environment Configuration
LEARNING_DB_URL=postgresql://voicestand:learning_pass@localhost:5433/voicestand_learning
GATEWAY_PORT=7890
GATEWAY_HOST=0.0.0.0

# Learning system configuration
ACCURACY_TARGET=0.95
LEARNING_RATE=0.001
UK_ENGLISH_SPECIALIZATION=true
ENSEMBLE_SIZE=5

# Monitoring
MONITOR_INTERVAL=10
ACCURACY_ALERTS=true
EOF

print_status "Environment configuration created ✅"

# Create a test script
print_header "Creating test script..."
cat > test_persistent_storage.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for VoiceStand PostgreSQL persistent storage
"""

import asyncio
import sys
import json
from datetime import datetime

# Add the gateway directory to path
sys.path.insert(0, 'learning/gateway')

from database import DatabaseOperations, RecognitionHistoryModel, init_database, cleanup_database

async def test_storage():
    """Test the persistent storage functionality"""
    print("🧪 Testing VoiceStand Persistent Storage...")

    try:
        # Initialize database
        await init_database()
        print("✅ Database connection established")

        # Test storing a recognition
        recognition = RecognitionHistoryModel(
            recognition_id="test_001",
            recognized_text="Hello, this is a test recognition",
            confidence=0.95,
            model_used="whisper_medium",
            processing_time_ms=150,
            is_uk_english=True,
            ground_truth="Hello, this is a test recognition"
        )

        await DatabaseOperations.store_recognition(recognition)
        print("✅ Recognition stored successfully")

        # Test retrieving metrics
        metrics = await DatabaseOperations.get_system_metrics()
        print(f"✅ Retrieved metrics: {len(metrics)} entries")

        # Test retrieving models
        models = await DatabaseOperations.get_model_performance()
        print(f"✅ Retrieved models: {len(models)} entries")

        # Test activity log
        await DatabaseOperations.add_activity(
            "test", "🧪 Test activity from storage verification"
        )
        activities = await DatabaseOperations.get_recent_activity(limit=5)
        print(f"✅ Activity log working: {len(activities)} entries")

        print("🎉 All storage tests passed!")

    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False

    finally:
        await cleanup_database()

    return True

if __name__ == "__main__":
    success = asyncio.run(test_storage())
    sys.exit(0 if success else 1)
EOF

chmod +x test_persistent_storage.py
print_status "Test script created ✅"

# Final summary
echo ""
print_header "🎉 PostgreSQL Persistent Storage Setup Complete!"
echo ""
print_status "Database URL: postgresql://voicestand:learning_pass@localhost:5433/voicestand_learning"
print_status "Gateway Port: 7890"
print_status "Dashboard: http://localhost:7890"
print_status "API Docs: http://localhost:7890/api/docs"
echo ""
print_header "Next steps:"
echo "1. Test the storage: ./test_persistent_storage.py"
echo "2. Start the learning gateway: cd learning/gateway && python main.py"
echo "3. Start the full learning system: $COMPOSE_CMD -f docker-compose.learning.yml up"
echo ""
print_header "Database management:"
echo "• Check status: cd learning/gateway && python migrate.py status"
echo "• Run migrations: cd learning/gateway && python migrate.py migrate"
echo "• Verify schema: cd learning/gateway && python migrate.py verify"
echo ""
print_status "Setup completed successfully! 🚀"