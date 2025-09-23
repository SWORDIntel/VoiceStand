#include "hotkey_manager.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <cctype>
#include <X11/Xutil.h>

namespace vtt {

HotkeyManager::HotkeyManager()
    : display_(nullptr)
    , root_window_(0)
    , is_running_(false)
    , hotkey_callback_(nullptr) {
}

HotkeyManager::~HotkeyManager() {
    cleanup();
}

bool HotkeyManager::initialize() {
    display_ = XOpenDisplay(nullptr);
    if (!display_) {
        std::cerr << "Failed to open X11 display\n";
        return false;
    }
    
    root_window_ = DefaultRootWindow(display_);
    
    return true;
}

bool HotkeyManager::register_hotkey(const std::string& hotkey_str) {
    unsigned int modifiers = 0;
    KeySym keysym = NoSymbol;

    if (!parse_hotkey_string(hotkey_str, modifiers, keysym)) {
        std::cerr << "Failed to parse hotkey string: " << hotkey_str << "\n";
        return false;
    }

    KeyCode keycode = XKeysymToKeycode(display_, keysym);
    if (keycode == 0) {
        std::cerr << "Failed to get keycode for keysym\n";
        return false;
    }

    // Grab key with error checking
    int result = XGrabKey(display_, keycode, modifiers, root_window_, False,
                         GrabModeAsync, GrabModeAsync);
    if (result == BadAccess) {
        std::cerr << "Warning: Failed to grab key - already grabbed by another application\n";
        return false;
    }

    // Grab with modifier variations (ignore errors for these as they're optional)
    XGrabKey(display_, keycode, modifiers | Mod2Mask, root_window_, False,
             GrabModeAsync, GrabModeAsync);
    XGrabKey(display_, keycode, modifiers | LockMask, root_window_, False,
             GrabModeAsync, GrabModeAsync);
    XGrabKey(display_, keycode, modifiers | Mod2Mask | LockMask, root_window_, False,
             GrabModeAsync, GrabModeAsync);

    // Store only in unified storage - remove legacy storage for new registrations
    InputBinding binding(InputType::Keyboard, modifiers, keycode, hotkey_str);
    input_bindings_[binding] = hotkey_str;

    XFlush(display_);

    return true;
}

bool HotkeyManager::register_mouse_button(const std::string& button_str) {
    unsigned int modifiers = 0;
    unsigned int button = 0;

    if (!parse_mouse_button_string(button_str, modifiers, button)) {
        std::cerr << "Failed to parse mouse button string: " << button_str << "\n";
        return false;
    }

    // Grab mouse button with error checking
    int result = XGrabButton(display_, button, modifiers, root_window_, False,
                            ButtonPressMask, GrabModeAsync, GrabModeAsync, None, None);
    if (result == BadAccess) {
        std::cerr << "Warning: Failed to grab mouse button - already grabbed by another application\n";
        return false;
    }

    // Also grab with modifier variations (ignore errors for these as they're optional)
    XGrabButton(display_, button, modifiers | Mod2Mask, root_window_, False,
                ButtonPressMask, GrabModeAsync, GrabModeAsync, None, None);
    XGrabButton(display_, button, modifiers | LockMask, root_window_, False,
                ButtonPressMask, GrabModeAsync, GrabModeAsync, None, None);
    XGrabButton(display_, button, modifiers | Mod2Mask | LockMask, root_window_, False,
                ButtonPressMask, GrabModeAsync, GrabModeAsync, None, None);

    InputBinding binding(InputType::Mouse, modifiers, button, button_str);
    input_bindings_[binding] = button_str;

    XFlush(display_);

    std::cout << "[INFO] Registered mouse button: " << button_str << " (button " << button << ")\n";
    return true;
}

void HotkeyManager::unregister_all_hotkeys() {
    // Unregister legacy keyboard hotkeys
    for (const auto& [key, _] : hotkeys_) {
        XUngrabKey(display_, key.second, key.first, root_window_);
        XUngrabKey(display_, key.second, key.first | Mod2Mask, root_window_);
        XUngrabKey(display_, key.second, key.first | LockMask, root_window_);
        XUngrabKey(display_, key.second, key.first | Mod2Mask | LockMask, root_window_);
    }

    // Unregister unified input bindings
    for (const auto& [binding, _] : input_bindings_) {
        if (binding.type == InputType::Keyboard) {
            XUngrabKey(display_, binding.keycode, binding.modifiers, root_window_);
            XUngrabKey(display_, binding.keycode, binding.modifiers | Mod2Mask, root_window_);
            XUngrabKey(display_, binding.keycode, binding.modifiers | LockMask, root_window_);
            XUngrabKey(display_, binding.keycode, binding.modifiers | Mod2Mask | LockMask, root_window_);
        } else if (binding.type == InputType::Mouse) {
            XUngrabButton(display_, binding.button, binding.modifiers, root_window_);
            XUngrabButton(display_, binding.button, binding.modifiers | Mod2Mask, root_window_);
            XUngrabButton(display_, binding.button, binding.modifiers | LockMask, root_window_);
            XUngrabButton(display_, binding.button, binding.modifiers | Mod2Mask | LockMask, root_window_);
        }
    }

    hotkeys_.clear();
    input_bindings_.clear();
    XFlush(display_);
}

void HotkeyManager::start() {
    if (is_running_) {
        return;
    }
    
    is_running_ = true;
    event_thread_ = std::thread(&HotkeyManager::event_loop, this);
}

void HotkeyManager::stop() {
    if (!is_running_) {
        return;
    }
    
    is_running_ = false;
    
    XEvent event;
    memset(&event, 0, sizeof(event));
    event.type = ClientMessage;
    event.xclient.window = root_window_;
    event.xclient.format = 32;
    XSendEvent(display_, root_window_, False, SubstructureNotifyMask, &event);
    XFlush(display_);
    
    if (event_thread_.joinable()) {
        event_thread_.join();
    }
}

void HotkeyManager::set_hotkey_callback(HotkeyCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    hotkey_callback_ = callback;
}

void HotkeyManager::cleanup() {
    stop();
    unregister_all_hotkeys();
    
    if (display_) {
        XCloseDisplay(display_);
        display_ = nullptr;
    }
}

void HotkeyManager::event_loop() {
    XEvent event;

    while (is_running_) {
        // Non-blocking check for events to allow clean shutdown
        if (XPending(display_) == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (!is_running_) break;  // Check again before blocking call
        XNextEvent(display_, &event);

        if (event.type == KeyPress) {
            unsigned int clean_modifiers = clean_modifier_mask(event.xkey.state);

            // Check unified input bindings only
            InputBinding binding(InputType::Keyboard, clean_modifiers, event.xkey.keycode, "");
            auto binding_it = input_bindings_.find(binding);
            if (binding_it != input_bindings_.end()) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (hotkey_callback_) {
                    hotkey_callback_(binding_it->second);
                }
            }
        }
        else if (event.type == ButtonPress) {
            unsigned int clean_modifiers = clean_modifier_mask(event.xbutton.state);

            // Check mouse button bindings
            InputBinding binding(InputType::Mouse, clean_modifiers, event.xbutton.button, "");
            auto binding_it = input_bindings_.find(binding);
            if (binding_it != input_bindings_.end()) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (hotkey_callback_) {
                    hotkey_callback_(binding_it->second);
                }
            }
        }
    }
}

bool HotkeyManager::parse_hotkey_string(const std::string& hotkey_str,
                                       unsigned int& modifiers, KeySym& keysym) {
    modifiers = 0;
    keysym = NoSymbol;
    
    std::string str = hotkey_str;
    size_t pos = 0;
    
    while ((pos = str.find('+')) != std::string::npos) {
        std::string mod = str.substr(0, pos);
        
        if (mod == "Ctrl" || mod == "Control") {
            modifiers |= ControlMask;
        } else if (mod == "Alt") {
            modifiers |= Mod1Mask;
        } else if (mod == "Shift") {
            modifiers |= ShiftMask;
        } else if (mod == "Super" || mod == "Win" || mod == "Meta") {
            modifiers |= Mod4Mask;
        }
        
        str.erase(0, pos + 1);
    }
    
    if (!str.empty()) {
        keysym = XStringToKeysym(str.c_str());
        if (keysym == NoSymbol) {
            if (str == "Space") {
                keysym = XK_space;
            } else if (str == "Return" || str == "Enter") {
                keysym = XK_Return;
            } else if (str == "Tab") {
                keysym = XK_Tab;
            } else if (str == "Escape" || str == "Esc") {
                keysym = XK_Escape;
            } else if (str.length() == 1) {
                keysym = XStringToKeysym(str.c_str());
            }
        }
    }
    
    return keysym != NoSymbol;
}

bool HotkeyManager::parse_mouse_button_string(const std::string& button_str,
                                             unsigned int& modifiers, unsigned int& button) {
    modifiers = 0;
    button = 0;

    std::string str = button_str;
    size_t pos = 0;

    // Parse modifiers
    while ((pos = str.find('+')) != std::string::npos) {
        std::string mod = str.substr(0, pos);

        if (mod == "Ctrl" || mod == "Control") {
            modifiers |= ControlMask;
        } else if (mod == "Alt") {
            modifiers |= Mod1Mask;
        } else if (mod == "Shift") {
            modifiers |= ShiftMask;
        } else if (mod == "Super" || mod == "Win" || mod == "Meta") {
            modifiers |= Mod4Mask;
        }

        str.erase(0, pos + 1);
    }

    // Parse mouse button
    if (!str.empty()) {
        if (str == "Left" || str == "Button1" || str == "LeftClick") {
            button = Button1;
        } else if (str == "Middle" || str == "Button2" || str == "MiddleClick") {
            button = Button2;
        } else if (str == "Right" || str == "Button3" || str == "RightClick") {
            button = Button3;
        } else if (str == "ScrollUp" || str == "Button4" || str == "WheelUp") {
            button = Button4;
        } else if (str == "ScrollDown" || str == "Button5" || str == "WheelDown") {
            button = Button5;
        } else if (str == "Back" || str == "Button8" || str == "Mouse4") {
            button = 8;  // Side button (back)
        } else if (str == "Forward" || str == "Button9" || str == "Mouse5") {
            button = 9;  // Side button (forward)
        } else if (str == "ScrollLeft" || str == "Button6") {
            button = 6;  // Horizontal scroll left
        } else if (str == "ScrollRight" || str == "Button7") {
            button = 7;  // Horizontal scroll right
        } else if (str == "Extra1" || str == "Button10") {
            button = 10; // Extra button 1
        } else if (str == "Extra2" || str == "Button11") {
            button = 11; // Extra button 2
        } else if (str == "Extra3" || str == "Button12") {
            button = 12; // Extra button 3
        } else if (str == "Extra4" || str == "Button13") {
            button = 13; // Extra button 4
        } else if (str == "Extra5" || str == "Button14") {
            button = 14; // Extra button 5
        } else if (str == "Extra6" || str == "Button15") {
            button = 15; // Extra button 6
        } else {
            // Try to parse as number with robust validation
            try {
                // Validate input string first
                if (str.empty() || str.length() > 3) {
                    return false;  // Reject empty or overly long strings
                }

                // Check for non-digit characters
                for (char c : str) {
                    if (!std::isdigit(c)) {
                        return false;
                    }
                }

                size_t pos = 0;
                int btn_num = std::stoi(str, &pos);

                // Ensure entire string was consumed
                if (pos != str.length()) {
                    return false;
                }

                if (btn_num >= 1 && btn_num <= 31) { // X11 supports up to 31 buttons
                    button = static_cast<unsigned int>(btn_num);
                } else {
                    return false;
                }
            } catch (const std::invalid_argument&) {
                return false;
            } catch (const std::out_of_range&) {
                return false;
            } catch (...) {
                return false;
            }
        }
    }

    return button != 0;
}

unsigned int HotkeyManager::clean_modifier_mask(unsigned int state) {
    return state & (ShiftMask | ControlMask | Mod1Mask | Mod4Mask);
}

}