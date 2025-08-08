#pragma once

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <string>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>

namespace vtt {

using HotkeyCallback = std::function<void(const std::string&)>;

class HotkeyManager {
public:
    HotkeyManager();
    ~HotkeyManager();
    
    bool initialize();
    bool register_hotkey(const std::string& hotkey_str);
    void unregister_all_hotkeys();
    
    void start();
    void stop();
    
    void set_hotkey_callback(HotkeyCallback callback);
    
private:
    void cleanup();
    void event_loop();
    bool parse_hotkey_string(const std::string& hotkey_str, 
                           unsigned int& modifiers, KeySym& keysym);
    unsigned int clean_modifier_mask(unsigned int state);
    
    Display* display_;
    Window root_window_;
    
    std::map<std::pair<unsigned int, KeyCode>, std::string> hotkeys_;
    
    std::atomic<bool> is_running_;
    std::thread event_thread_;
    
    HotkeyCallback hotkey_callback_;
    std::mutex callback_mutex_;
};

}