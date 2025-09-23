#include "src/integration/hotkey_manager.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <thread>

using namespace vtt;

void print_usage() {
    std::cout << "Mouse Button Test for VoiceStand\n";
    std::cout << "================================\n";
    std::cout << "Supported mouse button formats:\n";
    std::cout << "  Basic buttons: Left, Middle, Right\n";
    std::cout << "  Scroll wheel: ScrollUp, ScrollDown\n";
    std::cout << "  Side buttons: Back, Forward\n";
    std::cout << "  Numeric: Button1, Button2, Button3, etc.\n";
    std::cout << "  With modifiers: Ctrl+Right, Alt+Middle, Shift+Left\n";
    std::cout << "\nExamples:\n";
    std::cout << "  Right              - Right mouse button\n";
    std::cout << "  Ctrl+Middle        - Ctrl + middle mouse button\n";
    std::cout << "  Alt+Back           - Alt + side button (back)\n";
    std::cout << "  Shift+ScrollUp     - Shift + scroll wheel up\n";
    std::cout << "\nPress Ctrl+C to exit\n\n";
}

int main(int /* argc */, char* /* argv */[]) {
    print_usage();

    HotkeyManager manager;
    if (!manager.initialize()) {
        std::cerr << "Failed to initialize hotkey manager\n";
        return 1;
    }

    // Set up callback to handle button presses
    manager.set_hotkey_callback([](const std::string& binding) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S")
                  << "] Triggered: " << binding << std::endl;
    });

    // Register some example mouse button bindings
    std::vector<std::string> test_bindings = {
        "Right",           // Right click
        "Middle",          // Middle click
        "Ctrl+Left",       // Ctrl + left click
        "Alt+Right",       // Alt + right click
        "Shift+Middle",    // Shift + middle click
        "Back",            // Side button (back)
        "Forward",         // Side button (forward)
        "ScrollUp",        // Scroll wheel up
        "ScrollDown",      // Scroll wheel down
        "Ctrl+ScrollUp",   // Ctrl + scroll wheel up
    };

    std::cout << "Registering mouse button bindings:\n";
    for (const auto& binding : test_bindings) {
        if (manager.register_mouse_button(binding)) {
            std::cout << "  ✓ " << binding << std::endl;
        } else {
            std::cout << "  ✗ " << binding << " (failed)" << std::endl;
        }
    }

    // Also register a keyboard hotkey for comparison
    if (manager.register_hotkey("Ctrl+Alt+T")) {
        std::cout << "  ✓ Ctrl+Alt+T (keyboard)" << std::endl;
    }

    std::cout << "\nStarting event loop...\n";
    std::cout << "Try the registered mouse buttons and keyboard hotkeys!\n\n";

    manager.start();

    // Keep the program running
    try {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& e) {
        std::cout << "\nShutting down...\n";
    }

    manager.stop();
    return 0;
}