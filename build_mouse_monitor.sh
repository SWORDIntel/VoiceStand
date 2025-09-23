#!/bin/bash

# Quick build script for mouse button monitor
# This builds just the monitor tool without full VoiceStand dependencies

echo "Building VoiceStand Mouse Button Monitor..."

# Compile the monitor
g++ -std=c++17 -Wall -Wextra -O2 \
    -I/usr/include/X11 \
    mouse_button_monitor.cpp \
    -lX11 \
    -o mouse-button-monitor

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Usage:"
    echo "  ./mouse-button-monitor"
    echo ""
    echo "The monitor will show:"
    echo "  - Button numbers for all mouse buttons"
    echo "  - Modifier key combinations"
    echo "  - Binding strings for VoiceStand configuration"
    echo "  - Raw X11 event data"
    echo ""
    echo "Press Ctrl+C to exit the monitor."
else
    echo "✗ Build failed!"
    echo "Make sure you have X11 development headers installed:"
    echo "  Ubuntu/Debian: sudo apt install libx11-dev"
    echo "  Fedora/RHEL: sudo dnf install libX11-devel"
    echo "  Arch: sudo pacman -S libx11"
    exit 1
fi