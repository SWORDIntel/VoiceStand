#!/bin/bash

# NPU Build Script for VoiceStand Intel Integration
# Compiles NPU-accelerated voice-to-text system with Intel hardware optimization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PROJECT_NAME="VoiceStand NPU Integration"
TARGET_DIR="target/release"
NPU_CLI_NAME="npu-vtt"
RUST_LOG_LEVEL="info"
BUILD_FEATURES="npu,gna,simd-avx2"

print_status "Building $PROJECT_NAME"
print_status "Features: $BUILD_FEATURES"

# Check prerequisites
print_status "Checking prerequisites..."

# Check Rust installation
if ! command -v rustc &> /dev/null; then
    print_error "Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

RUST_VERSION=$(rustc --version)
print_status "Rust version: $RUST_VERSION"

# Check Cargo
if ! command -v cargo &> /dev/null; then
    print_error "Cargo is not available"
    exit 1
fi

# Check for Intel hardware (optional)
if lscpu | grep -q "Intel"; then
    CPU_INFO=$(lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//')
    print_status "Intel CPU detected: $CPU_INFO"

    # Check for Meteor Lake NPU specifically
    if lscpu | grep -q "Ultra 7"; then
        print_success "Intel Meteor Lake CPU detected - NPU likely available"
    else
        print_warning "NPU may not be available on this CPU"
    fi
else
    print_warning "Non-Intel CPU detected - NPU features will not be available"
fi

# Check OpenVINO installation
print_status "Checking OpenVINO installation..."
if command -v python3 &> /dev/null; then
    if python3 -c "import openvino" 2>/dev/null; then
        OPENVINO_VERSION=$(python3 -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "unknown")
        print_success "OpenVINO detected: version $OPENVINO_VERSION"
    else
        print_warning "OpenVINO Python package not found - NPU features may be limited"
        print_status "Install with: pip install openvino"
    fi
else
    print_warning "Python3 not available for OpenVINO check"
fi

# Check system dependencies
print_status "Checking system dependencies..."

MISSING_DEPS=()

# Check for PulseAudio development headers
if ! pkg-config --exists libpulse; then
    MISSING_DEPS+=("libpulse-dev")
fi

# Check for ALSA development headers
if ! pkg-config --exists alsa; then
    MISSING_DEPS+=("libasound2-dev")
fi

# Check for GTK4 development headers
if ! pkg-config --exists gtk4; then
    MISSING_DEPS+=("libgtk-4-dev")
fi

# Check for JSON development headers
if ! pkg-config --exists jsoncpp; then
    MISSING_DEPS+=("libjsoncpp-dev")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_warning "Missing system dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  - $dep"
    done
    print_status "Install with: sudo apt install ${MISSING_DEPS[*]}"

    read -p "Continue build anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "All system dependencies are available"
fi

# Create necessary directories
print_status "Creating build directories..."
mkdir -p "$TARGET_DIR"
mkdir -p "models"
mkdir -p "logs"

# Set environment variables
export RUST_LOG="$RUST_LOG_LEVEL"
export RUST_BACKTRACE="1"

# Build configuration
print_status "Configuring build..."

# Check if we're in release or debug mode
BUILD_MODE="release"
BUILD_FLAGS="--release"

if [[ "$1" == "debug" ]]; then
    BUILD_MODE="debug"
    BUILD_FLAGS=""
    TARGET_DIR="target/debug"
    print_status "Building in debug mode"
else
    print_status "Building in release mode"
fi

# Set optimization flags for Intel hardware
if [[ "$BUILD_FEATURES" == *"simd-avx2"* ]]; then
    export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma"
    print_status "Enabled AVX2/FMA optimizations"
fi

# Build Rust workspace
print_status "Building Rust workspace..."
print_status "Command: cargo build $BUILD_FLAGS --features=\"$BUILD_FEATURES\""

if cargo build $BUILD_FLAGS --features="$BUILD_FEATURES"; then
    print_success "Rust workspace build completed"
else
    print_error "Rust workspace build failed"
    exit 1
fi

# Build NPU CLI tool
print_status "Building NPU CLI tool..."

# Create CLI binary
if cargo build $BUILD_FLAGS --bin npu-cli --features="$BUILD_FEATURES"; then
    print_success "NPU CLI build completed"

    # Copy and rename the binary
    if [ -f "$TARGET_DIR/npu-cli" ]; then
        cp "$TARGET_DIR/npu-cli" "$TARGET_DIR/$NPU_CLI_NAME"
        chmod +x "$TARGET_DIR/$NPU_CLI_NAME"
        print_success "NPU CLI tool available as: $TARGET_DIR/$NPU_CLI_NAME"
    fi
else
    print_warning "NPU CLI build failed, but continuing with main build"
fi

# Build C++ components
print_status "Building C++ components..."

if [ -f "CMakeLists.txt" ]; then
    # Create build directory for CMake
    mkdir -p build
    cd build

    # Configure with CMake
    print_status "Configuring CMake..."
    if cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_NPU=ON .. ; then
        print_success "CMake configuration successful"

        # Build with make
        print_status "Building with make..."
        if make -j$(nproc); then
            print_success "C++ build completed"
        else
            print_error "C++ build failed"
            cd ..
            exit 1
        fi
    else
        print_error "CMake configuration failed"
        cd ..
        exit 1
    fi

    cd ..
else
    print_warning "CMakeLists.txt not found, skipping C++ build"
fi

# Run basic tests
print_status "Running basic tests..."

# Test NPU hardware detection (if available)
if [ -f "$TARGET_DIR/$NPU_CLI_NAME" ]; then
    print_status "Testing NPU hardware detection..."
    if "$TARGET_DIR/$NPU_CLI_NAME" status 2>/dev/null; then
        print_success "NPU hardware test passed"
    else
        print_warning "NPU hardware test failed (expected if no NPU hardware)"
    fi
fi

# Test Rust components
print_status "Testing Rust components..."
if cargo test --features="$BUILD_FEATURES" --lib 2>/dev/null; then
    print_success "Rust tests passed"
else
    print_warning "Some Rust tests failed"
fi

# Generate build information
print_status "Generating build information..."

BUILD_INFO_FILE="$TARGET_DIR/build-info.txt"
cat > "$BUILD_INFO_FILE" << EOF
VoiceStand NPU Integration Build Information
==========================================
Build Date: $(date)
Build Mode: $BUILD_MODE
Features: $BUILD_FEATURES
Rust Version: $(rustc --version)
Cargo Version: $(cargo --version)
System: $(uname -a)
CPU: $(lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//' || echo "Unknown")
EOF

if command -v python3 &> /dev/null && python3 -c "import openvino" 2>/dev/null; then
    echo "OpenVINO Version: $(python3 -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "unknown")" >> "$BUILD_INFO_FILE"
fi

print_success "Build information saved to: $BUILD_INFO_FILE"

# Display build summary
print_status "Build Summary"
print_status "============="

if [ -f "$TARGET_DIR/voice-to-text" ]; then
    MAIN_SIZE=$(du -h "$TARGET_DIR/voice-to-text" | cut -f1)
    print_success "Main VoiceStand binary: $TARGET_DIR/voice-to-text ($MAIN_SIZE)"
fi

if [ -f "$TARGET_DIR/$NPU_CLI_NAME" ]; then
    CLI_SIZE=$(du -h "$TARGET_DIR/$NPU_CLI_NAME" | cut -f1)
    print_success "NPU CLI tool: $TARGET_DIR/$NPU_CLI_NAME ($CLI_SIZE)"
fi

if [ -f "build/voice-to-text" ]; then
    CPP_SIZE=$(du -h "build/voice-to-text" | cut -f1)
    print_success "C++ VoiceStand binary: build/voice-to-text ($CPP_SIZE)"
fi

# Usage instructions
print_status ""
print_status "Usage Instructions"
print_status "=================="
print_status "NPU Status Check:"
print_status "  $TARGET_DIR/$NPU_CLI_NAME status --detailed"
print_status ""
print_status "Model Compilation:"
print_status "  $TARGET_DIR/$NPU_CLI_NAME compile --input model.onnx --precision fp16"
print_status ""
print_status "Voice Transcription:"
print_status "  $TARGET_DIR/$NPU_CLI_NAME listen --model whisper-base-npu.xml --hotkey \"ctrl+alt+space\""
print_status ""
print_status "Performance Benchmarks:"
print_status "  $TARGET_DIR/$NPU_CLI_NAME benchmark --comprehensive --detailed"
print_status ""

# Performance recommendations
print_status "Performance Recommendations"
print_status "=========================="
print_status "1. Run on Intel Meteor Lake CPU with NPU for optimal performance"
print_status "2. Use FP16 precision for best balance of speed and accuracy"
print_status "3. Enable power save mode for battery-powered devices"
print_status "4. Use wake word detection to minimize power consumption"
print_status ""

# Check for potential issues
print_status "Checking for potential issues..."

# Check available disk space
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 1048576 ]; then  # Less than 1GB
    print_warning "Low disk space available ($(($AVAILABLE_SPACE / 1024))MB)"
    print_status "Consider cleaning up disk space for optimal model caching"
fi

# Check memory
if [ -f "/proc/meminfo" ]; then
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    if [ "$TOTAL_MEM" -lt 8388608 ]; then  # Less than 8GB
        print_warning "Less than 8GB RAM available"
        print_status "Consider reducing model size or batch size for optimal performance"
    fi
fi

print_success "Build completed successfully!"
print_status "For detailed NPU integration documentation, see: docs/NPU_INTEGRATION.md"

# Exit successfully
exit 0