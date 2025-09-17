#!/bin/bash

# VoiceStand Rust Build Script
# Memory-safe voice-to-text system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦀 VoiceStand Rust Build System${NC}"
echo "================================================"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Cargo not found. Please install Rust:${NC}"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check if GTK4 development libraries are installed
check_gtk4() {
    echo -e "${BLUE}🔍 Checking GTK4 dependencies...${NC}"

    if ! pkg-config --exists gtk4; then
        echo -e "${YELLOW}⚠️  GTK4 development libraries not found${NC}"
        echo "Please install GTK4 development packages:"
        echo ""
        echo "Ubuntu/Debian:"
        echo "  sudo apt install libgtk-4-dev build-essential"
        echo ""
        echo "Fedora:"
        echo "  sudo dnf install gtk4-devel gcc"
        echo ""
        echo "Arch Linux:"
        echo "  sudo pacman -S gtk4 base-devel"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✅ GTK4 development libraries found${NC}"
    fi
}

# Check audio dependencies
check_audio_deps() {
    echo -e "${BLUE}🔍 Checking audio dependencies...${NC}"

    if ! pkg-config --exists alsa; then
        echo -e "${YELLOW}⚠️  ALSA development libraries not found${NC}"
        echo "Please install audio development packages:"
        echo ""
        echo "Ubuntu/Debian:"
        echo "  sudo apt install libasound2-dev"
        echo ""
        echo "Fedora:"
        echo "  sudo dnf install alsa-lib-devel"
        echo ""
        echo "Arch Linux:"
        echo "  sudo pacman -S alsa-lib"
    else
        echo -e "${GREEN}✅ Audio libraries found${NC}"
    fi
}

# Parse command line arguments
BUILD_TYPE="release"
VERBOSE=""
CLEAN=false
TEST=false
INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            TEST=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --help|-h)
            echo "VoiceStand Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug        Build debug version (default: release)"
            echo "  --release      Build release version"
            echo "  --verbose, -v  Verbose output"
            echo "  --clean        Clean before building"
            echo "  --test         Run tests after building"
            echo "  --install      Install to system"
            echo "  --help, -h     Show this help"
            echo ""
            echo "Dependencies checked:"
            echo "  - Rust toolchain"
            echo "  - GTK4 development libraries"
            echo "  - ALSA development libraries"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Show build configuration
echo -e "${BLUE}📋 Build Configuration:${NC}"
echo "  Build type: $BUILD_TYPE"
echo "  Verbose: ${VERBOSE:-disabled}"
echo "  Clean: $CLEAN"
echo "  Test: $TEST"
echo "  Install: $INSTALL"
echo ""

# Check dependencies
check_gtk4
check_audio_deps

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${BLUE}🧹 Cleaning previous build...${NC}"
    cargo clean $VERBOSE
fi

# Build the project
echo -e "${BLUE}🔨 Building VoiceStand...${NC}"

BUILD_CMD="cargo build"
if [ "$BUILD_TYPE" = "release" ]; then
    BUILD_CMD="$BUILD_CMD --release"
fi

if [ -n "$VERBOSE" ]; then
    BUILD_CMD="$BUILD_CMD $VERBOSE"
fi

# Set environment variables for cross-platform compilation
export PKG_CONFIG_ALLOW_CROSS=1

# Build with error handling
if $BUILD_CMD; then
    echo -e "${GREEN}✅ Build successful!${NC}"

    # Show binary location
    if [ "$BUILD_TYPE" = "release" ]; then
        BINARY_PATH="target/release/voicestand"
    else
        BINARY_PATH="target/debug/voicestand"
    fi

    if [ -f "$BINARY_PATH" ]; then
        echo -e "${GREEN}📦 Binary created: $BINARY_PATH${NC}"

        # Show binary size
        SIZE=$(du -h "$BINARY_PATH" | cut -f1)
        echo -e "${BLUE}📏 Binary size: $SIZE${NC}"
    fi
else
    echo -e "${RED}❌ Build failed!${NC}"
    echo ""
    echo "Common issues and solutions:"
    echo ""
    echo "1. Missing GTK4 development libraries:"
    echo "   Ubuntu/Debian: sudo apt install libgtk-4-dev"
    echo "   Fedora: sudo dnf install gtk4-devel"
    echo "   Arch: sudo pacman -S gtk4"
    echo ""
    echo "2. Missing audio libraries:"
    echo "   Ubuntu/Debian: sudo apt install libasound2-dev"
    echo "   Fedora: sudo dnf install alsa-lib-devel"
    echo "   Arch: sudo pacman -S alsa-lib"
    echo ""
    echo "3. Rust toolchain issues:"
    echo "   rustup update"
    echo ""
    exit 1
fi

# Run tests if requested
if [ "$TEST" = true ]; then
    echo -e "${BLUE}🧪 Running tests...${NC}"
    if cargo test $VERBOSE; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
    else
        echo -e "${RED}❌ Some tests failed${NC}"
        exit 1
    fi
fi

# Install if requested
if [ "$INSTALL" = true ]; then
    echo -e "${BLUE}📦 Installing VoiceStand...${NC}"

    # Install to ~/.local/bin by default
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"

    if cp "$BINARY_PATH" "$INSTALL_DIR/voicestand"; then
        echo -e "${GREEN}✅ Installed to $INSTALL_DIR/voicestand${NC}"
        echo ""
        echo "To run VoiceStand:"
        echo "  $INSTALL_DIR/voicestand"
        echo ""
        echo "Add to PATH (add to ~/.bashrc or ~/.zshrc):"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        echo -e "${RED}❌ Installation failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}🎉 VoiceStand build complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Download a speech model:"
echo "   ./target/release/voicestand --download-model base"
echo ""
echo "2. Run VoiceStand:"
echo "   ./target/release/voicestand"
echo ""
echo "3. Configure hotkeys and settings in the GUI"
echo ""
echo "For help:"
echo "   ./target/release/voicestand --help"