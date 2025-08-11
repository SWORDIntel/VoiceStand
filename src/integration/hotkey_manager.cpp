#include "hotkey_manager.h"
#include <iostream>
#include <cstring>
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
    
    XGrabKey(display_, keycode, modifiers, root_window_, False,
             GrabModeAsync, GrabModeAsync);
    
    XGrabKey(display_, keycode, modifiers | Mod2Mask, root_window_, False,
             GrabModeAsync, GrabModeAsync);
    XGrabKey(display_, keycode, modifiers | LockMask, root_window_, False,
             GrabModeAsync, GrabModeAsync);
    XGrabKey(display_, keycode, modifiers | Mod2Mask | LockMask, root_window_, False,
             GrabModeAsync, GrabModeAsync);
    
    hotkeys_[{modifiers, keycode}] = hotkey_str;
    
    XFlush(display_);
    
    return true;
}

void HotkeyManager::unregister_all_hotkeys() {
    for (const auto& [key, _] : hotkeys_) {
        XUngrabKey(display_, key.second, key.first, root_window_);
        XUngrabKey(display_, key.second, key.first | Mod2Mask, root_window_);
        XUngrabKey(display_, key.second, key.first | LockMask, root_window_);
        XUngrabKey(display_, key.second, key.first | Mod2Mask | LockMask, root_window_);
    }
    
    hotkeys_.clear();
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
        XNextEvent(display_, &event);
        
        if (event.type == KeyPress) {
            unsigned int clean_modifiers = clean_modifier_mask(event.xkey.state);
            
            auto it = hotkeys_.find({clean_modifiers, event.xkey.keycode});
            if (it != hotkeys_.end()) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (hotkey_callback_) {
                    hotkey_callback_(it->second);
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

unsigned int HotkeyManager::clean_modifier_mask(unsigned int state) {
    return state & (ShiftMask | ControlMask | Mod1Mask | Mod4Mask);
}

}