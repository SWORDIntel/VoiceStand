#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cstring>
#include <vector>
#include <map>
#include <sstream>
#include <csignal>

class MouseButtonMonitor {
private:
    Display* display_;
    Window root_window_;
    bool running_;

    // Known button mappings
    std::map<int, std::string> button_names_ = {
        {1, "Left (Button1)"},
        {2, "Middle (Button2)"},
        {3, "Right (Button3)"},
        {4, "ScrollUp (Button4)"},
        {5, "ScrollDown (Button5)"},
        {6, "ScrollLeft (Button6)"},
        {7, "ScrollRight (Button7)"},
        {8, "Back/Mouse4 (Button8)"},
        {9, "Forward/Mouse5 (Button9)"},
        {10, "Extra1 (Button10)"},
        {11, "Extra2 (Button11)"},
        {12, "Extra3 (Button12)"},
        {13, "Extra4 (Button13)"},
        {14, "Extra5 (Button14)"},
        {15, "Extra6 (Button15)"}
    };

public:
    MouseButtonMonitor() : display_(nullptr), root_window_(0), running_(false) {}

    ~MouseButtonMonitor() {
        cleanup();
    }

    bool initialize() {
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            std::cerr << "Failed to open X11 display\n";
            return false;
        }

        root_window_ = DefaultRootWindow(display_);

        // Select mouse button events on root window with error checking
        int result = XSelectInput(display_, root_window_,
                                 ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
        if (result == BadWindow) {
            std::cerr << "Failed to select input events on root window\n";
            XCloseDisplay(display_);
            display_ = nullptr;
            return false;
        }

        return true;
    }

    void print_header() {
        std::cout << "\n=== VoiceStand Mouse Button Monitor ===\n";
        std::cout << "Press any mouse button to see its details\n";
        std::cout << "Use Ctrl+C to exit\n";
        std::cout << "----------------------------------------\n";
        std::cout << std::left
                  << std::setw(12) << "Time"
                  << std::setw(15) << "Event"
                  << std::setw(8) << "Button"
                  << std::setw(25) << "Description"
                  << std::setw(20) << "Modifiers"
                  << std::setw(15) << "Coordinates"
                  << "\n";
        std::cout << std::string(95, '-') << "\n";
    }

    std::string get_modifier_string(unsigned int state) {
        std::vector<std::string> mods;

        if (state & ControlMask) mods.push_back("Ctrl");
        if (state & Mod1Mask) mods.push_back("Alt");
        if (state & ShiftMask) mods.push_back("Shift");
        if (state & Mod4Mask) mods.push_back("Super");
        if (state & Mod2Mask) mods.push_back("NumLock");
        if (state & LockMask) mods.push_back("CapsLock");

        if (mods.empty()) return "None";

        std::string result;
        for (size_t i = 0; i < mods.size(); ++i) {
            if (i > 0) result += "+";
            result += mods[i];
        }
        return result;
    }

    std::string get_time_string() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        return ss.str();
    }

    std::string get_button_description(int button) {
        auto it = button_names_.find(button);
        if (it != button_names_.end()) {
            return it->second;
        }
        return "Unknown (" + std::to_string(button) + ")";
    }

    std::string generate_binding_string(int button, unsigned int state) {
        std::vector<std::string> parts;

        // Add modifiers (clean mask)
        unsigned int clean_state = state & (ControlMask | Mod1Mask | ShiftMask | Mod4Mask);
        if (clean_state & ControlMask) parts.push_back("Ctrl");
        if (clean_state & Mod1Mask) parts.push_back("Alt");
        if (clean_state & ShiftMask) parts.push_back("Shift");
        if (clean_state & Mod4Mask) parts.push_back("Super");

        // Add button name
        std::string button_name;
        switch (button) {
            case 1: button_name = "Left"; break;
            case 2: button_name = "Middle"; break;
            case 3: button_name = "Right"; break;
            case 4: button_name = "ScrollUp"; break;
            case 5: button_name = "ScrollDown"; break;
            case 6: button_name = "ScrollLeft"; break;
            case 7: button_name = "ScrollRight"; break;
            case 8: button_name = "Back"; break;
            case 9: button_name = "Forward"; break;
            default: button_name = "Button" + std::to_string(button); break;
        }

        parts.push_back(button_name);

        // Join with '+'
        std::string result;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) result += "+";
            result += parts[i];
        }

        return result;
    }

    void monitor() {
        if (!display_) {
            std::cerr << "Display not initialized\n";
            return;
        }

        print_header();
        running_ = true;

        XEvent event;
        while (running_) {
            // Non-blocking check for events to allow clean shutdown
            if (XPending(display_) == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (!running_) break;  // Check again before blocking call
            XNextEvent(display_, &event);

            if (event.type == ButtonPress) {
                std::string time_str = get_time_string();
                std::string event_type = "PRESS";
                int button = event.xbutton.button;
                std::string description = get_button_description(button);
                std::string modifiers = get_modifier_string(event.xbutton.state);
                std::string coords = "(" + std::to_string(event.xbutton.x_root) +
                                   "," + std::to_string(event.xbutton.y_root) + ")";

                std::cout << std::left
                          << std::setw(12) << time_str
                          << std::setw(15) << event_type
                          << std::setw(8) << button
                          << std::setw(25) << description
                          << std::setw(20) << modifiers
                          << std::setw(15) << coords
                          << "\n";

                // Show suggested binding string
                std::string binding = generate_binding_string(button, event.xbutton.state);
                std::cout << "    → Binding string: \"" << binding << "\"\n";

                // Show raw hex values for debugging
                std::cout << "    → Raw: button=" << button
                          << " state=0x" << std::hex << event.xbutton.state << std::dec
                          << "\n\n";
            }
            else if (event.type == ButtonRelease) {
                // Optionally show release events (commented out to reduce noise)
                /*
                std::string time_str = get_time_string();
                std::cout << std::left
                          << std::setw(12) << time_str
                          << std::setw(15) << "RELEASE"
                          << std::setw(8) << event.xbutton.button
                          << "\n";
                */
            }
        }
    }

    void stop() {
        running_ = false;
    }

private:
    void cleanup() {
        if (display_) {
            XCloseDisplay(display_);
            display_ = nullptr;
        }
    }
};

// Global monitor instance for signal handling
MouseButtonMonitor* g_monitor = nullptr;

void signal_handler(int /* sig */) {
    std::cout << "\n\nShutting down mouse monitor...\n";
    if (g_monitor) {
        g_monitor->stop();
    }
}

int main() {
    std::cout << "VoiceStand Mouse Button Discovery Tool\n";
    std::cout << "======================================\n";

    MouseButtonMonitor monitor;
    g_monitor = &monitor;

    // Set up signal handler for clean exit
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!monitor.initialize()) {
        std::cerr << "Failed to initialize mouse monitor\n";
        return 1;
    }

    std::cout << "\nInstructions:\n";
    std::cout << "1. Press any mouse button to see its button number\n";
    std::cout << "2. Try combining with Ctrl, Alt, Shift, or Super (Windows key)\n";
    std::cout << "3. Note the \"Binding string\" output for VoiceStand configuration\n";
    std::cout << "4. Test scroll wheel, side buttons, and any extra buttons\n";
    std::cout << "5. Press Ctrl+C when done\n";

    monitor.monitor();

    std::cout << "Mouse monitoring stopped.\n";
    return 0;
}