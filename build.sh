#!/bin/bash

set -e

echo "🎤 Building Standalone Voice-to-Text System"
echo "==========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
THIRD_PARTY_DIR="${PROJECT_DIR}/third_party"

echo "📁 Setting up directories..."
mkdir -p "${BUILD_DIR}"
mkdir -p "${THIRD_PARTY_DIR}"

echo "📦 Checking dependencies..."
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        echo "   Run: sudo apt-get install $2"
        exit 1
    fi
}

check_package() {
    if ! pkg-config --exists "$1"; then
        echo "❌ $1 is not installed. Please install it first."
        echo "   Run: sudo apt-get install $2"
        exit 1
    fi
}

check_dependency "cmake" "cmake"
check_dependency "g++" "g++"
check_dependency "pkg-config" "pkg-config"
check_dependency "wget" "wget"
check_package "gtk4" "libgtk-4-dev"
check_package "libpulse" "libpulse-dev"
check_package "jsoncpp" "libjsoncpp-dev"

if ! pkg-config --exists x11; then
    echo "📦 Installing X11 development libraries..."
    sudo apt-get update
    sudo apt-get install -y libx11-dev
fi

echo "🔧 Building whisper.cpp..."
if [ ! -d "${THIRD_PARTY_DIR}/whisper.cpp" ]; then
    echo "📥 Cloning whisper.cpp repository..."
    git clone https://github.com/ggerganov/whisper.cpp.git "${THIRD_PARTY_DIR}/whisper.cpp"
fi

cd "${THIRD_PARTY_DIR}/whisper.cpp"

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake .. -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF
make -j$(nproc)

echo "🔨 Building Voice-to-Text application..."
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_INCLUDE_DIR="${THIRD_PARTY_DIR}/whisper.cpp" \
    -DWHISPER_LIB_DIR="${THIRD_PARTY_DIR}/whisper.cpp/build"

make -j$(nproc)

echo ""
echo "✅ Build complete!"
echo ""
echo "📦 Downloading default model (if not present)..."

MODEL_DIR="${PROJECT_DIR}/models"
mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_DIR}/ggml-base.bin" ]; then
    echo "📥 Downloading Whisper base model..."
    wget -P "${MODEL_DIR}" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
else
    echo "✓ Model already downloaded"
fi

echo ""
echo "🎉 Setup complete! You can now run the application:"
echo ""
echo "   ${BUILD_DIR}/voice-to-text"
echo ""
echo "Or download a different model size:"
echo "   ${BUILD_DIR}/voice-to-text --download-model <tiny|base|small|medium|large>"
echo ""
echo "Press Ctrl+Alt+Space to toggle recording (default hotkey)"

chmod +x "${BUILD_DIR}/voice-to-text" 2>/dev/null || true