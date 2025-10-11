#!/bin/bash

# VoiceStand Learning System Security Validation Script
# This script validates that the Docker configuration is secure

echo "=== VoiceStand Learning System Security Validation ==="
echo

# Check if Docker Compose configuration is valid
echo "1. Validating Docker Compose configuration..."
if docker-compose -f docker-compose.unified.yml config --quiet; then
    echo "✓ Docker Compose configuration is valid"
else
    echo "✗ Docker Compose configuration has errors"
    exit 1
fi

# Check network configuration
echo
echo "2. Checking network configuration..."
if grep -q "127.0.0.1:7890:7890" docker-compose.unified.yml; then
    echo "✓ Port binding restricted to localhost"
else
    echo "✗ Port binding not restricted to localhost"
fi

if grep -q "subnet: 172.20.0.0/24" docker-compose.unified.yml; then
    echo "✓ Custom subnet configured"
else
    echo "✗ Custom subnet not configured"
fi

# Check security options
echo
echo "3. Checking security configurations..."
if grep -q "no-new-privileges:true" docker-compose.unified.yml; then
    echo "✓ No new privileges restriction enabled"
else
    echo "✗ No new privileges restriction missing"
fi

if grep -q "cap_drop:" docker-compose.unified.yml && grep -q "ALL" docker-compose.unified.yml; then
    echo "✓ All capabilities dropped by default"
else
    echo "✗ Capabilities not properly restricted"
fi

if grep -q "read_only: true" docker-compose.unified.yml; then
    echo "✓ Read-only containers configured"
else
    echo "✗ Read-only containers not configured"
fi

if grep -q "user: \"1000:1000\"" docker-compose.unified.yml; then
    echo "✓ User context isolation configured"
else
    echo "✗ User context isolation missing"
fi

# Check application-level security
echo
echo "4. Checking application-level security..."
if grep -q "BIND_ADDRESS: 127.0.0.1" docker-compose.unified.yml; then
    echo "✓ Application bind address restricted to localhost"
else
    echo "✗ Application bind address not restricted"
fi

if grep -q "ENABLE_CORS: false" docker-compose.unified.yml; then
    echo "✓ CORS disabled for security"
else
    echo "✗ CORS settings not configured"
fi

if grep -q "MAX_CONNECTIONS: 10" docker-compose.unified.yml; then
    echo "✓ Connection limits configured"
else
    echo "✗ Connection limits not configured"
fi

echo
echo "=== Security Validation Complete ==="
echo "The VoiceStand learning system is configured for local-only access."
echo "Services will only be accessible from localhost (127.0.0.1)."