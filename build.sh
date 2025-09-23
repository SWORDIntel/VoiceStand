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
echo "🎯 VoiceStand Model Selection"
echo "=============================="

MODEL_DIR="${PROJECT_DIR}/models"
mkdir -p "${MODEL_DIR}"

# Function to detect hardware capabilities
detect_hardware() {
    local ram_gb
    local cpu_info
    local has_npu=false
    local has_meteor_lake=false

    # Detect RAM (in GB)
    ram_gb=$(free -g | awk '/^Mem:/{print $2}')

    # Detect CPU information
    cpu_info=$(lscpu | grep "Model name" || echo "Unknown CPU")

    # Check for Intel NPU/Meteor Lake indicators
    if lscpu | grep -qi "meteor\|ultra" || [ -d "/sys/class/intel_npu" ] 2>/dev/null; then
        has_meteor_lake=true
        has_npu=true
    fi

    echo "🔍 Hardware Detection Results:"
    echo "   CPU: $(echo "$cpu_info" | cut -d':' -f2 | xargs)"
    echo "   RAM: ${ram_gb}GB available"
    echo "   Intel NPU: $([ "$has_npu" = true ] && echo "✓ Detected" || echo "❌ Not detected")"
    echo "   Meteor Lake: $([ "$has_meteor_lake" = true ] && echo "✓ Detected" || echo "❌ Not detected")"
    echo ""

    # Store results for recommendation logic
    echo "$ram_gb $has_npu $has_meteor_lake"
}

# Function to recommend model based on hardware
recommend_model() {
    local hardware_info
    local ram_gb has_npu has_meteor_lake

    hardware_info=$(detect_hardware)
    read -r ram_gb has_npu has_meteor_lake <<< "$hardware_info"

    if [ "$has_meteor_lake" = true ] && [ "$ram_gb" -ge 8 ]; then
        echo "small"  # Professional use with NPU acceleration
    elif [ "$has_meteor_lake" = true ] && [ "$ram_gb" -ge 4 ]; then
        echo "base"   # Balanced for NPU systems
    elif [ "$ram_gb" -ge 6 ]; then
        echo "small"  # Good performance on high-RAM systems
    elif [ "$ram_gb" -ge 4 ]; then
        echo "base"   # Standard for most systems
    else
        echo "tiny"   # Conservative for low-RAM systems
    fi
}

# Function to show model information
show_model_info() {
    cat << 'EOF'

📊 Available Whisper Models:

┌─────────┬──────────┬───────────┬─────────────┬─────────────┬──────────┬────────────────────┐
│ Model   │ Size     │ RAM Usage │ NPU Latency │ Studio/Clear│ WER      │ Best For           │
├─────────┼──────────┼───────────┼─────────────┼─────────────┼──────────┼────────────────────┤
│ tiny    │ 39MB     │ ~390MB    │ <1ms        │ 70%/65%     │ 32-40%   │ Development/Testing│
│ base    │ 142MB    │ ~500MB    │ <2ms        │ 80%/75%     │ 25-32%   │ Recommended Default│
│ small   │ 244MB    │ ~1GB      │ <3ms        │ 84%/79%     │ 20-28%   │ Professional Use   │
│ medium  │ 769MB    │ ~2GB      │ <5ms        │ 90%/85%     │ 15-22%   │ High Accuracy      │
│ large   │ 1550MB   │ ~4-6GB    │ <8ms        │ 93%/88%     │ 12-18%   │ Maximum Accuracy   │
└─────────┴──────────┴───────────┴─────────────┴─────────────┴──────────┴────────────────────┘

📝 Accuracy Guide:
   • Studio/Clear = Accuracy for high-quality vs. normal speech
   • WER = Word Error Rate (lower is better)
   • Noisy environments: Reduce accuracy by ~10-15%
   • Phone calls/video: Reduce accuracy by ~5-10%

EOF
}

# Function to get user choice with recommendations
get_model_choice() {
    local recommended_model
    local choice

    recommended_model=$(recommend_model)

    show_model_info

    echo "🎯 Based on your hardware, we recommend: $recommended_model"
    echo ""
    echo "Model Selection Options:"
    echo "  1) tiny    - 39MB, 70%/65% accuracy, <1ms NPU (testing/development)"
    echo "  2) base    - 142MB, 80%/75% accuracy, <2ms NPU (recommended default)"
    echo "  3) small   - 244MB, 84%/79% accuracy, <3ms NPU (professional use)"
    echo "  4) medium  - 769MB, 90%/85% accuracy, <5ms NPU (high accuracy)"
    echo "  5) large   - 1550MB, 93%/88% accuracy, <8ms NPU (maximum accuracy)"
    echo "  6) Skip    - Download manually later"
    echo ""

    while true; do
        read -p "Choose model [1-6] or press Enter for recommended ($recommended_model): " choice

        case $choice in
            ""|"")
                echo "$recommended_model"
                return
                ;;
            1) echo "tiny"; return ;;
            2) echo "base"; return ;;
            3) echo "small"; return ;;
            4) echo "medium"; return ;;
            5) echo "large"; return ;;
            6) echo "skip"; return ;;
            *) echo "Invalid choice. Please enter 1-6 or press Enter." ;;
        esac
    done
}

# Function to show download progress with model info
download_model_with_progress() {
    local model_size="$1"
    local model_file="${MODEL_DIR}/ggml-${model_size}.bin"

    # Model URLs and metadata
    declare -A model_urls=(
        ["tiny"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
        ["base"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
        ["small"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
        ["medium"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
        ["large"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
    )

    declare -A model_sizes=(
        ["tiny"]="39MB"
        ["base"]="142MB"
        ["small"]="244MB"
        ["medium"]="769MB"
        ["large"]="1550MB"
    )

    echo "📥 Downloading $model_size model (${model_sizes[$model_size]})..."
    echo "🔗 Source: ${model_urls[$model_size]}"
    echo ""

    # Download with progress bar
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$model_file" "${model_urls[$model_size]}"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$model_file" "${model_urls[$model_size]}"
    else
        echo "❌ Neither wget nor curl is available for downloading"
        return 1
    fi

    # Verify download
    if [ -f "$model_file" ] && [ -s "$model_file" ]; then
        echo "✅ Model downloaded successfully: $model_file"

        # Create a default configuration for this model
        create_model_config "$model_size"

        return 0
    else
        echo "❌ Download failed or file is empty"
        rm -f "$model_file"
        return 1
    fi
}

# Function to create model-specific configuration
create_model_config() {
    local model_size="$1"
    local config_dir="${HOME}/.config/voice-to-text"
    local config_file="${config_dir}/config.json"

    mkdir -p "$config_dir"

    # Model-specific optimizations
    case $model_size in
        "tiny")
            cat > "$config_file" << EOF
{
  "model": {
    "name": "tiny",
    "path": "${MODEL_DIR}/ggml-tiny.bin",
    "threads": 1,
    "memory_pool_mb": 256
  },
  "audio": {
    "sample_rate": 16000,
    "vad_threshold": 0.6,
    "buffer_duration_ms": 500
  },
  "hardware": {
    "prefer_npu": true,
    "cpu_fallback": true
  }
}
EOF
            ;;
        "base")
            cat > "$config_file" << EOF
{
  "model": {
    "name": "base",
    "path": "${MODEL_DIR}/ggml-base.bin",
    "threads": 2,
    "memory_pool_mb": 512
  },
  "audio": {
    "sample_rate": 16000,
    "vad_threshold": 0.5,
    "buffer_duration_ms": 1000
  },
  "hardware": {
    "prefer_npu": true,
    "cpu_fallback": true
  }
}
EOF
            ;;
        "small"|"medium"|"large")
            local threads=4
            local memory_mb=1024
            [ "$model_size" = "medium" ] && memory_mb=2048
            [ "$model_size" = "large" ] && memory_mb=6144 && threads=6

            cat > "$config_file" << EOF
{
  "model": {
    "name": "$model_size",
    "path": "${MODEL_DIR}/ggml-${model_size}.bin",
    "threads": $threads,
    "memory_pool_mb": $memory_mb
  },
  "audio": {
    "sample_rate": 16000,
    "vad_threshold": 0.4,
    "buffer_duration_ms": 1500
  },
  "hardware": {
    "prefer_npu": true,
    "cpu_fallback": true,
    "numa_aware": true
  }
}
EOF
            ;;
    esac

    echo "⚙️  Configuration created: $config_file"
}

# Main model selection logic
main_model_selection() {
    local choice
    local existing_models

    # Check for existing models
    existing_models=$(find "$MODEL_DIR" -name "ggml-*.bin" 2>/dev/null | wc -l)

    if [ "$existing_models" -gt 0 ]; then
        echo "📦 Found $existing_models existing model(s) in $MODEL_DIR"
        echo ""
        find "$MODEL_DIR" -name "ggml-*.bin" -exec basename {} \; | sed 's/^/   ✓ /'
        echo ""

        read -p "Download additional model? [y/N]: " choice
        if [[ ! "$choice" =~ ^[Yy] ]]; then
            echo "✓ Using existing models"
            return 0
        fi
    fi

    # Get user's model choice
    choice=$(get_model_choice)

    if [ "$choice" = "skip" ]; then
        echo "⏭️  Skipping model download. You can download later with:"
        echo "   ${BUILD_DIR}/voice-to-text --download-model <model-size>"
        return 0
    fi

    # Download the selected model
    if download_model_with_progress "$choice"; then
        echo ""
        echo "🎉 Model setup complete!"
        echo "📍 Model location: ${MODEL_DIR}/ggml-${choice}.bin"
        echo "⚙️  Configuration: ~/.config/voice-to-text/config.json"
    else
        echo ""
        echo "❌ Model download failed. You can try again later with:"
        echo "   ${BUILD_DIR}/voice-to-text --download-model $choice"
        return 1
    fi
}

# Execute model selection
main_model_selection

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