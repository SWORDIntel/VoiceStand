#pragma once

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <string>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <stdexcept>

namespace vtt {

using HotkeyCallback = std::function<void(const std::string&)>;

// Input binding types
enum class InputType {
    Keyboard,
    Mouse
};

struct InputBinding {
    InputType type;
    unsigned int modifiers;
    union {
        KeyCode keycode;     // For keyboard
        unsigned int button; // For mouse (1=left, 2=middle, 3=right, 4=scroll_up, 5=scroll_down, 8=back, 9=forward)
    };
    std::string binding_string;

    InputBinding(InputType t, unsigned int mod, KeyCode key, const std::string& str)
        : type(t), modifiers(mod), binding_string(str) {
        if (t != InputType::Keyboard) {
            throw std::invalid_argument("KeyCode constructor requires InputType::Keyboard");
        }
        keycode = key;
    }

    InputBinding(InputType t, unsigned int mod, unsigned int btn, const std::string& str)
        : type(t), modifiers(mod), binding_string(str) {
        if (t != InputType::Mouse) {
            throw std::invalid_argument("Button constructor requires InputType::Mouse");
        }
        button = btn;
    }

    bool operator<(const InputBinding& other) const {
        if (type != other.type) return type < other.type;
        if (modifiers != other.modifiers) return modifiers < other.modifiers;
        if (type == InputType::Keyboard) {
            return keycode < other.keycode;
        } else {
            return button < other.button;
        }
    }
};

class HotkeyManager {
public:
    HotkeyManager();
    ~HotkeyManager();

    bool initialize();
    bool register_hotkey(const std::string& hotkey_str);
    bool register_mouse_button(const std::string& button_str);
    void unregister_all_hotkeys();

    void start();
    void stop();

    void set_hotkey_callback(HotkeyCallback callback);

private:
    void cleanup();
    void event_loop();
    bool parse_hotkey_string(const std::string& hotkey_str,
                           unsigned int& modifiers, KeySym& keysym);
    bool parse_mouse_button_string(const std::string& button_str,
                                 unsigned int& modifiers, unsigned int& button);
    unsigned int clean_modifier_mask(unsigned int state);

    Display* display_;
    Window root_window_;

    // Legacy keyboard hotkey storage (for backward compatibility)
    std::map<std::pair<unsigned int, KeyCode>, std::string> hotkeys_;

    // New unified input binding storage
    std::map<InputBinding, std::string> input_bindings_;

    std::atomic<bool> is_running_;
    std::thread event_thread_;

    HotkeyCallback hotkey_callback_;
    std::mutex callback_mutex_;
};

}