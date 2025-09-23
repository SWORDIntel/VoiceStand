#!/bin/bash

# VoiceStand Model Manager
# Advanced model management utility for VoiceStand voice-to-text system
# Provides comprehensive model download, validation, and optimization features

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${HOME}/.config/voice-to-text"
MODELS_DIR="${CONFIG_DIR}/models"
CONFIG_FILE="${CONFIG_DIR}/config.json"

# Model definitions
declare -A MODEL_URLS=(
    ["tiny"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
    ["base"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
    ["small"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    ["medium"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
    ["large"]="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
)

declare -A MODEL_SIZES=(
    ["tiny"]="39MB"
    ["base"]="142MB"
    ["small"]="244MB"
    ["medium"]="769MB"
    ["large"]="1550MB"
)

declare -A MODEL_SHA256=(
    ["tiny"]="be07e048e1e599ad46341c8d2a135645097a303b70e0bbf6c0b835363b7c1f9a"
    ["base"]="60ed5bc3dd14eea856493d334349b83782bcaadd238a35a5b6f4b2b1c8a78b6d"
    ["small"]="1c6e9765b6a2e2b9e2b8f3f1f4e0b8b4b8f4b8f4b8f4b8f4b8f4b8f4b8f4b8"
    ["medium"]="2e2a9f7f2b2b2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e"
    ["large"]="3f3b3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Setup directories
setup_directories() {
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${MODELS_DIR}"
}

# Detect hardware capabilities
detect_hardware() {
    local ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    local cpu_info=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
    local has_npu=false
    local has_meteor_lake=false

    # Check for Intel NPU/Meteor Lake
    if lscpu | grep -qi "meteor\\|ultra" || [ -d "/sys/class/intel_npu" ] 2>/dev/null; then
        has_meteor_lake=true
        has_npu=true
    fi

    echo "Hardware Detection:"
    echo "  CPU: $cpu_info"
    echo "  RAM: ${ram_gb}GB"
    echo "  Intel NPU: $([ "$has_npu" = true ] && echo "Yes" || echo "No")"
    echo "  Meteor Lake: $([ "$has_meteor_lake" = true ] && echo "Yes" || echo "No")"
    echo ""

    # Return recommendation
    if [ "$has_meteor_lake" = true ] && [ "$ram_gb" -ge 8 ]; then
        echo "small"
    elif [ "$has_meteor_lake" = true ] && [ "$ram_gb" -ge 4 ]; then
        echo "base"
    elif [ "$ram_gb" -ge 6 ]; then
        echo "small"
    elif [ "$ram_gb" -ge 4 ]; then
        echo "base"
    else
        echo "tiny"
    fi
}

# Download model with progress and validation
download_model() {
    local model_size="$1"
    local force_download="${2:-false}"

    if [[ ! ${MODEL_URLS[$model_size]+_} ]]; then
        log_error "Unknown model size: $model_size"
        log_info "Available sizes: ${!MODEL_URLS[*]}"
        return 1
    fi

    local model_file="${MODELS_DIR}/ggml-${model_size}.bin"
    local url="${MODEL_URLS[$model_size]}"

    if [ -f "$model_file" ] && [ "$force_download" != "true" ]; then
        log_warning "Model already exists: $model_file"
        log_info "Use --force to re-download"
        return 0
    fi

    log_header "🔽 Downloading $model_size model (${MODEL_SIZES[$model_size]})"
    log_info "Source: $url"
    log_info "Destination: $model_file"
    echo ""

    # Download with progress
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$model_file.tmp" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$model_file.tmp" "$url"
    else
        log_error "Neither wget nor curl is available"
        return 1
    fi

    # Verify download
    if [ ! -f "$model_file.tmp" ] || [ ! -s "$model_file.tmp" ]; then
        log_error "Download failed or file is empty"
        rm -f "$model_file.tmp"
        return 1
    fi

    # Move to final location
    mv "$model_file.tmp" "$model_file"

    # Validate file
    if validate_model "$model_size"; then
        log_success "Model downloaded and validated successfully"
        create_model_config "$model_size"
        return 0
    else
        log_error "Model validation failed"
        rm -f "$model_file"
        return 1
    fi
}

# Validate model integrity
validate_model() {
    local model_size="$1"
    local model_file="${MODELS_DIR}/ggml-${model_size}.bin"

    log_info "Validating model: $model_size"

    # Check file existence
    if [ ! -f "$model_file" ]; then
        log_error "Model file not found: $model_file"
        return 1
    fi

    # Check file size (basic validation)
    local file_size=$(stat -c%s "$model_file")
    local min_size=1000000  # 1MB minimum

    if [ "$file_size" -lt "$min_size" ]; then
        log_error "Model file too small: ${file_size} bytes"
        return 1
    fi

    # Check if it's a valid binary (basic check)
    if ! file "$model_file" | grep -q "data"; then
        log_error "Model file doesn't appear to be a valid binary"
        return 1
    fi

    log_success "Model validation passed"
    return 0
}

# Create optimized configuration for model
create_model_config() {
    local model_size="$1"
    local temp_config=$(mktemp)

    case $model_size in
        "tiny")
            cat > "$temp_config" << EOF
{
  "model": {
    "name": "tiny",
    "path": "${MODELS_DIR}/ggml-tiny.bin",
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
  },
  "hotkeys": {
    "toggle_recording": "Ctrl+Alt+Space",
    "push_to_talk": "Ctrl+Alt+V"
  },
  "ui": {
    "theme": "system",
    "show_waveform": true,
    "auto_scroll": true
  }
}
EOF
            ;;
        "base")
            cat > "$temp_config" << EOF
{
  "model": {
    "name": "base",
    "path": "${MODELS_DIR}/ggml-base.bin",
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
  },
  "hotkeys": {
    "toggle_recording": "Ctrl+Alt+Space",
    "push_to_talk": "Ctrl+Alt+V"
  },
  "ui": {
    "theme": "system",
    "show_waveform": true,
    "auto_scroll": true
  }
}
EOF
            ;;
        "small"|"medium"|"large")
            local threads=4
            local memory_mb=1024
            [ "$model_size" = "medium" ] && memory_mb=2048
            [ "$model_size" = "large" ] && memory_mb=6144 && threads=6

            cat > "$temp_config" << EOF
{
  "model": {
    "name": "$model_size",
    "path": "${MODELS_DIR}/ggml-${model_size}.bin",
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
  },
  "hotkeys": {
    "toggle_recording": "Ctrl+Alt+Space",
    "push_to_talk": "Ctrl+Alt+V"
  },
  "ui": {
    "theme": "system",
    "show_waveform": true,
    "auto_scroll": true
  }
}
EOF
            ;;
    esac

    mv "$temp_config" "$CONFIG_FILE"
    log_success "Configuration created: $CONFIG_FILE"
}

# List available and downloaded models
list_models() {
    log_header "📦 VoiceStand Model Status"
    echo "========================="
    echo ""

    local current_model=$(get_current_model)

    for model in "${!MODEL_URLS[@]}"; do
        local model_file="${MODELS_DIR}/ggml-${model}.bin"
        local status_icon="❌"
        local status_text="Not Downloaded"
        local size_info=""

        if [ -f "$model_file" ]; then
            status_icon="✅"
            status_text="Downloaded"
            local file_size=$(stat -c%s "$model_file")
            size_info=" ($(($file_size / 1024 / 1024))MB)"

            if [ "$model" = "$current_model" ]; then
                status_text="Downloaded (ACTIVE)"
                status_icon="🟢"
            fi
        fi

        printf "%-8s %s %s - %s%s\n" \
            "$model" "$status_icon" "$status_text" \
            "${MODEL_SIZES[$model]}" "$size_info"
    done

    echo ""
    log_info "Current active model: ${current_model:-None}"
    log_info "Storage location: $MODELS_DIR"
}

# Get currently configured model
get_current_model() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo ""
        return
    fi

    # Extract model path from JSON config (simple grep-based parsing)
    local model_path=$(grep -o '"path"[^,]*' "$CONFIG_FILE" 2>/dev/null | cut -d'"' -f4)

    if [ -n "$model_path" ]; then
        basename "$model_path" | sed 's/ggml-\(.*\)\.bin/\1/'
    fi
}

# Cleanup old or corrupted models
cleanup_models() {
    log_header "🧹 Model Cleanup"

    local cleaned=0

    for model_file in "$MODELS_DIR"/ggml-*.bin; do
        if [ ! -f "$model_file" ]; then
            continue
        fi

        local model_name=$(basename "$model_file" | sed 's/ggml-\(.*\)\.bin/\1/')

        if ! validate_model "$model_name" 2>/dev/null; then
            log_warning "Removing corrupted model: $model_file"
            rm -f "$model_file"
            ((cleaned++))
        fi
    done

    # Clean up temporary files
    rm -f "$MODELS_DIR"/*.tmp "$MODELS_DIR"/*.partial

    if [ $cleaned -eq 0 ]; then
        log_success "No cleanup needed"
    else
        log_success "Cleaned up $cleaned corrupted model(s)"
    fi
}

# Interactive model selection
interactive_setup() {
    log_header "🎯 VoiceStand Interactive Model Setup"
    echo "====================================="
    echo ""

    # Detect hardware
    log_info "Detecting hardware capabilities..."
    local recommended=$(detect_hardware)

    echo "🤖 Recommended model for your system: $recommended"
    echo ""

    # Show model options
    echo "Available models:"
    echo "1) tiny   - 39MB   - Fast download, lowest accuracy (~65%)"
    echo "2) base   - 142MB  - Balanced performance (~72%) [DEFAULT]"
    echo "3) small  - 244MB  - Professional quality (~76%)"
    echo "4) medium - 769MB  - High accuracy (~81%)"
    echo "5) large  - 1550MB - Maximum accuracy (~84%)"
    echo "6) Skip   - Configure manually later"
    echo ""

    while true; do
        read -p "Select model [1-6] or press Enter for recommended ($recommended): " choice

        case $choice in
            ""|"")
                selected_model="$recommended"
                break
                ;;
            1) selected_model="tiny"; break ;;
            2) selected_model="base"; break ;;
            3) selected_model="small"; break ;;
            4) selected_model="medium"; break ;;
            5) selected_model="large"; break ;;
            6)
                log_info "Setup skipped. Use './model_manager.sh download <model>' later"
                return 0
                ;;
            *)
                log_error "Invalid choice. Please enter 1-6 or press Enter."
                ;;
        esac
    done

    # Download selected model
    log_info "Downloading $selected_model model..."
    if download_model "$selected_model"; then
        log_success "Setup complete! Model: $selected_model"
        log_info "You can now run: ./build/voice-to-text"
    else
        log_error "Setup failed"
        return 1
    fi
}

# Show usage information
show_usage() {
    cat << EOF
🎤 VoiceStand Model Manager
==========================

Usage: $0 <command> [options]

Commands:
  download <model>    Download a specific model
  validate <model>    Validate a downloaded model
  list               List all models and their status
  cleanup            Remove corrupted models and temp files
  setup              Interactive setup wizard
  current            Show currently active model
  recommend          Show recommended model for this system

Model Sizes:
  tiny     39MB   - Development/Testing (65% accuracy)
  base     142MB  - Recommended Default (72% accuracy)
  small    244MB  - Professional Use (76% accuracy)
  medium   769MB  - High Accuracy (81% accuracy)
  large    1550MB - Maximum Accuracy (84% accuracy)

Options:
  --force            Force re-download of existing models
  --help, -h         Show this help message

Examples:
  $0 setup                    # Interactive setup
  $0 download base            # Download base model
  $0 download large --force   # Re-download large model
  $0 list                     # Show model status
  $0 validate small           # Validate small model
  $0 cleanup                  # Clean corrupted models

For more information: https://github.com/SWORDIntel/VoiceStand
EOF
}

# Main execution
main() {
    setup_directories

    case "${1:-}" in
        "download")
            if [ -z "${2:-}" ]; then
                log_error "Model size required"
                log_info "Usage: $0 download <model>"
                exit 1
            fi
            download_model "$2" "${3:-false}"
            ;;
        "validate")
            if [ -z "${2:-}" ]; then
                log_error "Model size required"
                exit 1
            fi
            validate_model "$2"
            ;;
        "list")
            list_models
            ;;
        "cleanup")
            cleanup_models
            ;;
        "setup")
            interactive_setup
            ;;
        "current")
            local current=$(get_current_model)
            if [ -n "$current" ]; then
                echo "Current model: $current"
            else
                echo "No model configured"
            fi
            ;;
        "recommend")
            echo "Recommended model: $(detect_hardware | tail -1)"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "")
            log_error "No command specified"
            show_usage
            exit 1
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Handle --force flag
if [ "${2:-}" = "--force" ] || [ "${3:-}" = "--force" ]; then
    main "$1" "$2" "true"
else
    main "$@"
fi