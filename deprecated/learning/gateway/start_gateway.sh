#!/bin/bash

# VoiceStand Learning Gateway Startup Script
# Provides robust service management with health checks

set -e

GATEWAY_DIR="/home/john/GITHUB/VoiceStand/learning/gateway"
PIDFILE="$GATEWAY_DIR/gateway.pid"
LOGFILE="$GATEWAY_DIR/gateway.log"
PORT=7890

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌${NC} $1"
}

# Function to check if service is running
is_running() {
    if [ -f "$PIDFILE" ]; then
        local pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            rm -f "$PIDFILE"
            return 1
        fi
    fi
    return 1
}

# Function to check if port is available
check_port() {
    if netstat -tlnp 2>/dev/null | grep -q ":$PORT "; then
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local max_attempts=30
    local attempt=1

    print_status "Waiting for service to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            print_success "Service is ready and responding"
            return 0
        fi

        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "Service failed to start within 30 seconds"
    return 1
}

# Function to start the service
start_service() {
    print_status "Starting VoiceStand Learning Gateway..."

    # Check if already running
    if is_running; then
        print_warning "Service is already running (PID: $(cat $PIDFILE))"
        return 0
    fi

    # Check if port is available
    if ! check_port; then
        print_error "Port $PORT is already in use"
        print_status "Current port usage:"
        netstat -tlnp 2>/dev/null | grep ":$PORT "
        return 1
    fi

    # Change to gateway directory
    cd "$GATEWAY_DIR"

    # Start the service
    nohup python3 main.py > "$LOGFILE" 2>&1 &
    local pid=$!
    echo $pid > "$PIDFILE"

    # Wait for service to be ready
    if wait_for_service; then
        print_success "VoiceStand Learning Gateway started successfully"
        print_status "🌐 Dashboard: http://localhost:$PORT"
        print_status "📚 API Docs: http://localhost:$PORT/api/docs"
        print_status "📊 Health: http://localhost:$PORT/health"
        print_status "📝 Logs: tail -f $LOGFILE"
        return 0
    else
        # Service failed to start properly
        if [ -f "$PIDFILE" ]; then
            local pid=$(cat "$PIDFILE")
            kill "$pid" 2>/dev/null || true
            rm -f "$PIDFILE"
        fi
        print_error "Failed to start service"
        return 1
    fi
}

# Function to stop the service
stop_service() {
    print_status "Stopping VoiceStand Learning Gateway..."

    if ! is_running; then
        print_warning "Service is not running"
        return 0
    fi

    local pid=$(cat "$PIDFILE")
    kill "$pid" 2>/dev/null || true

    # Wait for process to stop
    local attempt=1
    while [ $attempt -le 10 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        attempt=$((attempt + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        print_warning "Force killing process..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PIDFILE"
    print_success "Service stopped"
}

# Function to restart the service
restart_service() {
    stop_service
    sleep 2
    start_service
}

# Function to show service status
status_service() {
    print_status "VoiceStand Learning Gateway Status:"

    if is_running; then
        local pid=$(cat "$PIDFILE")
        print_success "Service is running (PID: $pid)"

        # Test connectivity
        if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            print_success "Service is responding to requests"

            # Show service info
            echo ""
            print_status "Service Information:"
            curl -s http://localhost:$PORT/health | python3 -m json.tool 2>/dev/null || echo "Could not parse health response"
        else
            print_error "Service is running but not responding"
        fi
    else
        print_error "Service is not running"
    fi

    echo ""
    print_status "Port Status:"
    if netstat -tlnp 2>/dev/null | grep -q ":$PORT "; then
        netstat -tlnp 2>/dev/null | grep ":$PORT "
    else
        echo "Port $PORT is not in use"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOGFILE" ]; then
        print_status "Showing recent logs (press Ctrl+C to exit):"
        tail -f "$LOGFILE"
    else
        print_error "Log file not found: $LOGFILE"
    fi
}

# Function to test the service
test_service() {
    print_status "Testing VoiceStand Learning Gateway..."

    if ! is_running; then
        print_error "Service is not running"
        return 1
    fi

    local base_url="http://localhost:$PORT"
    local tests_passed=0
    local tests_total=0

    # Test health endpoint
    tests_total=$((tests_total + 1))
    if curl -s "$base_url/health" >/dev/null; then
        print_success "Health endpoint: OK"
        tests_passed=$((tests_passed + 1))
    else
        print_error "Health endpoint: FAILED"
    fi

    # Test dashboard
    tests_total=$((tests_total + 1))
    if curl -s "$base_url/" | grep -q "VoiceStand"; then
        print_success "Dashboard: OK"
        tests_passed=$((tests_passed + 1))
    else
        print_error "Dashboard: FAILED"
    fi

    # Test API endpoints
    for endpoint in "api/v1/metrics" "api/v1/models" "api/v1/dashboard-data"; do
        tests_total=$((tests_total + 1))
        if curl -s "$base_url/$endpoint" >/dev/null; then
            print_success "API $endpoint: OK"
            tests_passed=$((tests_passed + 1))
        else
            print_error "API $endpoint: FAILED"
        fi
    done

    echo ""
    print_status "Test Results: $tests_passed/$tests_total tests passed"

    if [ $tests_passed -eq $tests_total ]; then
        print_success "All tests passed! Service is fully operational"
        return 0
    else
        print_error "Some tests failed. Check service configuration"
        return 1
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        show_logs
        ;;
    test)
        test_service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the VoiceStand Learning Gateway"
        echo "  stop    - Stop the service"
        echo "  restart - Restart the service"
        echo "  status  - Show service status and connectivity"
        echo "  logs    - Show and follow service logs"
        echo "  test    - Run connectivity tests"
        echo ""
        echo "Quick access URLs:"
        echo "  Dashboard: http://localhost:$PORT"
        echo "  API Docs:  http://localhost:$PORT/api/docs"
        echo "  Health:    http://localhost:$PORT/health"
        exit 1
        ;;
esac