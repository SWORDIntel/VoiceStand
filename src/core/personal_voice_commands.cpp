#include "personal_voice_commands.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/extensions/XTest.h>

namespace vtt {

// Built-in system commands registration
void PersonalVoiceCommands::register_system_commands() {
    // Lock screen
    PersonalMacro lock_screen("lock_screen");
    lock_screen.voice_patterns = {
        "lock screen", "lock the screen", "lock my computer", "secure screen"
    };
    lock_screen.commands = {"loginctl lock-session"};
    lock_screen.privilege = PrivilegeLevel::SYSTEM;
    register_personal_macro(lock_screen);

    // Switch workspace
    PersonalMacro switch_workspace("switch_workspace");
    switch_workspace.voice_patterns = {
        "switch to workspace (.+)", "go to workspace (.+)", "workspace (.+)"
    };
    switch_workspace.commands = {"wmctrl -s {}"};
    switch_workspace.privilege = PrivilegeLevel::USER;
    register_personal_macro(switch_workspace);

    // Open terminal
    PersonalMacro open_terminal("open_terminal");
    open_terminal.voice_patterns = {
        "open terminal", "launch terminal", "open command line", "start terminal"
    };
    open_terminal.applications = {"gnome-terminal"};
    open_terminal.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_terminal);

    // Volume controls
    PersonalMacro volume_up("volume_up");
    volume_up.voice_patterns = {
        "volume up", "increase volume", "louder", "turn up volume"
    };
    volume_up.commands = {"pactl set-sink-volume @DEFAULT_SINK@ +5%"};
    volume_up.privilege = PrivilegeLevel::USER;
    register_personal_macro(volume_up);

    PersonalMacro volume_down("volume_down");
    volume_down.voice_patterns = {
        "volume down", "decrease volume", "quieter", "turn down volume"
    };
    volume_down.commands = {"pactl set-sink-volume @DEFAULT_SINK@ -5%"};
    volume_down.privilege = PrivilegeLevel::USER;
    register_personal_macro(volume_down);

    PersonalMacro volume_mute("volume_mute");
    volume_mute.voice_patterns = {
        "mute", "mute volume", "silence", "turn off sound"
    };
    volume_mute.commands = {"pactl set-sink-mute @DEFAULT_SINK@ toggle"};
    volume_mute.privilege = PrivilegeLevel::USER;
    register_personal_macro(volume_mute);

    // Screenshot
    PersonalMacro screenshot("screenshot");
    screenshot.voice_patterns = {
        "take screenshot", "screenshot", "capture screen", "take picture of screen"
    };
    screenshot.commands = {"gnome-screenshot"};
    screenshot.privilege = PrivilegeLevel::USER;
    register_personal_macro(screenshot);

    // System shutdown/restart (requires confirmation)
    PersonalMacro shutdown("shutdown");
    shutdown.voice_patterns = {
        "shutdown computer", "power off", "turn off computer"
    };
    shutdown.commands = {"systemctl poweroff"};
    shutdown.privilege = PrivilegeLevel::ADMIN;
    register_personal_macro(shutdown);

    PersonalMacro restart("restart");
    restart.voice_patterns = {
        "restart computer", "reboot", "restart system"
    };
    restart.commands = {"systemctl reboot"};
    restart.privilege = PrivilegeLevel::ADMIN;
    register_personal_macro(restart);
}

void PersonalVoiceCommands::register_application_commands() {
    // Web browser
    PersonalMacro open_browser("open_browser");
    open_browser.voice_patterns = {
        "open browser", "launch browser", "open firefox", "start browsing"
    };
    open_browser.applications = {"firefox"};
    open_browser.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_browser);

    // File manager
    PersonalMacro open_files("open_files");
    open_files.voice_patterns = {
        "open files", "file manager", "open file browser", "browse files"
    };
    open_files.applications = {"nautilus"};
    open_files.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_files);

    // Text editor
    PersonalMacro open_editor("open_editor");
    open_editor.voice_patterns = {
        "open editor", "text editor", "open gedit", "start editing"
    };
    open_editor.applications = {"gedit"};
    open_editor.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_editor);

    // Calculator
    PersonalMacro open_calculator("open_calculator");
    open_calculator.voice_patterns = {
        "open calculator", "calculator", "launch calculator"
    };
    open_calculator.applications = {"gnome-calculator"};
    open_calculator.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_calculator);

    // System monitor
    PersonalMacro open_monitor("open_monitor");
    open_monitor.voice_patterns = {
        "open system monitor", "task manager", "system monitor"
    };
    open_monitor.applications = {"gnome-system-monitor"};
    open_monitor.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_monitor);

    // Music player
    PersonalMacro open_music("open_music");
    open_music.voice_patterns = {
        "open music", "music player", "play music", "launch rhythmbox"
    };
    open_music.applications = {"rhythmbox"};
    open_music.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_music);
}

void PersonalVoiceCommands::register_window_management_commands() {
    // Minimize window
    PersonalMacro minimize_window("minimize_window");
    minimize_window.voice_patterns = {
        "minimize window", "minimize", "hide window"
    };
    minimize_window.keystrokes = {"Alt+F9"};
    minimize_window.privilege = PrivilegeLevel::USER;
    register_personal_macro(minimize_window);

    // Maximize window
    PersonalMacro maximize_window("maximize_window");
    maximize_window.voice_patterns = {
        "maximize window", "maximize", "full screen"
    };
    maximize_window.keystrokes = {"Alt+F10"};
    maximize_window.privilege = PrivilegeLevel::USER;
    register_personal_macro(maximize_window);

    // Close window
    PersonalMacro close_window("close_window");
    close_window.voice_patterns = {
        "close window", "close", "exit window"
    };
    close_window.keystrokes = {"Alt+F4"};
    close_window.privilege = PrivilegeLevel::USER;
    register_personal_macro(close_window);

    // Switch windows
    PersonalMacro switch_window("switch_window");
    switch_window.voice_patterns = {
        "switch window", "next window", "alt tab"
    };
    switch_window.keystrokes = {"Alt+Tab"};
    switch_window.privilege = PrivilegeLevel::USER;
    register_personal_macro(switch_window);

    // Move window left/right between monitors
    PersonalMacro move_window_left("move_window_left");
    move_window_left.voice_patterns = {
        "move window left", "window left screen"
    };
    move_window_left.keystrokes = {"Super+Shift+Left"};
    move_window_left.privilege = PrivilegeLevel::USER;
    register_personal_macro(move_window_left);

    PersonalMacro move_window_right("move_window_right");
    move_window_right.voice_patterns = {
        "move window right", "window right screen"
    };
    move_window_right.keystrokes = {"Super+Shift+Right"};
    move_window_right.privilege = PrivilegeLevel::USER;
    register_personal_macro(move_window_right);
}

void PersonalVoiceCommands::register_productivity_commands() {
    // Copy/paste operations
    PersonalMacro copy_text("copy_text");
    copy_text.voice_patterns = {
        "copy", "copy text", "copy this"
    };
    copy_text.keystrokes = {"Ctrl+c"};
    copy_text.privilege = PrivilegeLevel::USER;
    register_personal_macro(copy_text);

    PersonalMacro paste_text("paste_text");
    paste_text.voice_patterns = {
        "paste", "paste text", "paste this"
    };
    paste_text.keystrokes = {"Ctrl+v"};
    paste_text.privilege = PrivilegeLevel::USER;
    register_personal_macro(paste_text);

    // Save/open operations
    PersonalMacro save_file("save_file");
    save_file.voice_patterns = {
        "save", "save file", "save document"
    };
    save_file.keystrokes = {"Ctrl+s"};
    save_file.privilege = PrivilegeLevel::USER;
    register_personal_macro(save_file);

    PersonalMacro open_file("open_file");
    open_file.voice_patterns = {
        "open file", "open document"
    };
    open_file.keystrokes = {"Ctrl+o"};
    open_file.privilege = PrivilegeLevel::USER;
    register_personal_macro(open_file);

    // Undo/redo
    PersonalMacro undo("undo");
    undo.voice_patterns = {
        "undo", "undo last", "go back"
    };
    undo.keystrokes = {"Ctrl+z"};
    undo.privilege = PrivilegeLevel::USER;
    register_personal_macro(undo);

    PersonalMacro redo("redo");
    redo.voice_patterns = {
        "redo", "redo last", "go forward"
    };
    redo.keystrokes = {"Ctrl+y"};
    redo.privilege = PrivilegeLevel::USER;
    register_personal_macro(redo);

    // Find/replace
    PersonalMacro find_text("find_text");
    find_text.voice_patterns = {
        "find", "search", "find text"
    };
    find_text.keystrokes = {"Ctrl+f"};
    find_text.privilege = PrivilegeLevel::USER;
    register_personal_macro(find_text);

    // Quick productivity macros
    PersonalMacro new_tab("new_tab");
    new_tab.voice_patterns = {
        "new tab", "open new tab"
    };
    new_tab.keystrokes = {"Ctrl+t"};
    new_tab.privilege = PrivilegeLevel::USER;
    register_personal_macro(new_tab);

    PersonalMacro close_tab("close_tab");
    close_tab.voice_patterns = {
        "close tab", "close current tab"
    };
    close_tab.keystrokes = {"Ctrl+w"};
    close_tab.privilege = PrivilegeLevel::USER;
    register_personal_macro(close_tab);
}

bool PersonalVoiceCommands::register_personal_macro(const PersonalMacro& macro) {
    std::lock_guard<std::mutex> lock(macros_mutex_);

    auto it = personal_macros_.find(macro.name);
    if (it != personal_macros_.end()) {
        std::cout << "[WARN] Overwriting existing macro: " << macro.name << "\n";
    }

    personal_macros_[macro.name] = std::make_unique<PersonalMacro>(macro);
    std::cout << "[INFO] Registered personal macro: " << macro.name
              << " with " << macro.voice_patterns.size() << " patterns\n";
    return true;
}

bool PersonalVoiceCommands::remove_personal_macro(const std::string& name) {
    std::lock_guard<std::mutex> lock(macros_mutex_);

    auto it = personal_macros_.find(name);
    if (it == personal_macros_.end()) {
        return false;
    }

    personal_macros_.erase(it);
    std::cout << "[INFO] Removed personal macro: " << name << "\n";
    return true;
}

PersonalVoiceCommands::ExecutionResult PersonalVoiceCommands::execute_shell_commands(
    const PersonalMacro& macro, const CommandContext& context) {

    ExecutionResult result;
    result.command_name = macro.name;

    for (const auto& command : macro.commands) {
        std::string cmd = command;

        // Simple parameter substitution (for patterns with capture groups)
        // In a full implementation, you'd parse the regex matches properly
        if (cmd.find("{}") != std::string::npos) {
            // Extract parameter from voice input (simplified)
            std::regex param_regex(R"(\b(\w+)\s*$)");
            std::smatch matches;
            if (std::regex_search(context.raw_text, matches, param_regex)) {
                std::string param = matches[1].str();
                size_t pos = cmd.find("{}");
                cmd.replace(pos, 2, param);
            }
        }

        std::cout << "[EXEC] Running command: " << cmd << "\n";

        // Execute command and capture output
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            result.error_message = "Failed to execute command: " + cmd;
            result.exit_code = -1;
            return result;
        }

        char buffer[128];
        std::string command_output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            command_output += buffer;
        }

        int exit_status = pclose(pipe);
        result.exit_code = WEXITSTATUS(exit_status);
        result.output += command_output;

        if (result.exit_code != 0) {
            result.error_message = "Command failed with exit code: " + std::to_string(result.exit_code);
            return result;
        }
    }

    result.success = true;
    return result;
}

PersonalVoiceCommands::ExecutionResult PersonalVoiceCommands::execute_application_launch(
    const PersonalMacro& macro, const CommandContext& context) {

    ExecutionResult result;
    result.command_name = macro.name;

    for (const auto& app : macro.applications) {
        std::string command = app + " &";
        std::cout << "[EXEC] Launching application: " << app << "\n";

        int exit_code = system(command.c_str());

        if (exit_code != 0) {
            result.error_message = "Failed to launch application: " + app;
            result.exit_code = exit_code;
            return result;
        }

        result.output += "Launched: " + app + "\n";
    }

    result.success = true;
    return result;
}

PersonalVoiceCommands::ExecutionResult PersonalVoiceCommands::execute_keystroke_sequence(
    const PersonalMacro& macro, const CommandContext& context) {

    ExecutionResult result;
    result.command_name = macro.name;

    // Initialize X11 display
    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        result.error_message = "Cannot open X11 display for keystroke simulation";
        return result;
    }

    for (const auto& keystroke : macro.keystrokes) {
        std::cout << "[EXEC] Sending keystroke: " << keystroke << "\n";

        // Parse keystroke combination (e.g., "Ctrl+c", "Alt+Tab")
        std::vector<std::string> keys;
        std::stringstream ss(keystroke);
        std::string key;

        while (std::getline(ss, key, '+')) {
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            keys.push_back(key);
        }

        // Press modifier keys
        std::vector<KeyCode> pressed_keys;
        for (size_t i = 0; i < keys.size() - 1; ++i) {
            KeySym keysym = XStringToKeysym(keys[i].c_str());
            if (keysym == NoSymbol) {
                // Try common modifier mappings
                if (keys[i] == "Ctrl") keysym = XK_Control_L;
                else if (keys[i] == "Alt") keysym = XK_Alt_L;
                else if (keys[i] == "Shift") keysym = XK_Shift_L;
                else if (keys[i] == "Super") keysym = XK_Super_L;
            }

            if (keysym != NoSymbol) {
                KeyCode keycode = XKeysymToKeycode(display, keysym);
                if (keycode != 0) {
                    XTestFakeKeyEvent(display, keycode, True, 0);
                    pressed_keys.push_back(keycode);
                }
            }
        }

        // Press the main key
        if (!keys.empty()) {
            KeySym keysym = XStringToKeysym(keys.back().c_str());
            if (keysym != NoSymbol) {
                KeyCode keycode = XKeysymToKeycode(display, keysym);
                if (keycode != 0) {
                    XTestFakeKeyEvent(display, keycode, True, 0);
                    XTestFakeKeyEvent(display, keycode, False, 0);
                }
            }
        }

        // Release modifier keys in reverse order
        for (auto it = pressed_keys.rbegin(); it != pressed_keys.rend(); ++it) {
            XTestFakeKeyEvent(display, *it, False, 0);
        }

        XSync(display, False);
        usleep(10000);  // 10ms delay between keystrokes

        result.output += "Sent: " + keystroke + "\n";
    }

    XCloseDisplay(display);
    result.success = true;
    return result;
}

bool PersonalVoiceCommands::validate_command_security(const PersonalMacro& macro,
                                                      const CommandContext& context) {
    // Check privilege level
    if (macro.privilege > current_privilege_) {
        log_security_event(macro, context, "Insufficient privilege level");
        return false;
    }

    // Check if custom security validator is set
    if (security_validator_) {
        if (!security_validator_(macro, context)) {
            log_security_event(macro, context, "Custom security validator rejected command");
            return false;
        }
    }

    // Check confidence threshold for high-privilege commands
    if (macro.privilege >= PrivilegeLevel::SYSTEM && context.confidence < 0.90f) {
        log_security_event(macro, context, "Confidence too low for system command");
        return false;
    }

    return true;
}

bool PersonalVoiceCommands::request_user_confirmation(const PersonalMacro& macro,
                                                      const CommandContext& context) {
    std::cout << "\n[CONFIRMATION REQUIRED]\n";
    std::cout << "Command: " << macro.name << "\n";
    std::cout << "Privilege Level: " << static_cast<int>(macro.privilege) << "\n";
    std::cout << "Confidence: " << (context.confidence * 100) << "%\n";
    std::cout << "Execute this command? (y/N): ";

    std::string response;
    std::getline(std::cin, response);

    return (response == "y" || response == "Y" || response == "yes");
}

void PersonalVoiceCommands::log_security_event(const PersonalMacro& macro,
                                               const CommandContext& context,
                                               const std::string& reason) {
    // Log security events for audit purposes
    std::cout << "[SECURITY] Command rejected: " << macro.name
              << " - Reason: " << reason
              << " - Confidence: " << context.confidence << "\n";
}

void PersonalVoiceCommands::learn_from_execution(const std::string& command_name,
                                                 bool success, float confidence) {
    auto& pattern_info = pattern_learning_[command_name];
    pattern_info.usage_count++;

    if (success) {
        pattern_info.success_rate = (pattern_info.success_rate * (pattern_info.usage_count - 1) + 1.0f)
                                   / pattern_info.usage_count;
    } else {
        pattern_info.success_rate = (pattern_info.success_rate * (pattern_info.usage_count - 1))
                                   / pattern_info.usage_count;
    }
}

PersonalVoiceCommands::CommandStats PersonalVoiceCommands::get_statistics() const {
    return stats_;
}

void PersonalVoiceCommands::print_usage_report() const {
    std::cout << "\n=== Personal Voice Commands Usage Report ===\n";
    std::cout << "Total commands executed: " << stats_.total_commands_executed << "\n";
    std::cout << "Successful executions: " << stats_.successful_executions << "\n";
    std::cout << "Failed executions: " << stats_.failed_executions << "\n";
    std::cout << "Security rejections: " << stats_.security_rejections << "\n";
    std::cout << "Average confidence: " << (stats_.average_confidence * 100) << "%\n";
    std::cout << "Average execution time: " << stats_.average_execution_time_ms << " ms\n";

    std::cout << "\nMost used commands:\n";
    for (const auto& [command, count] : stats_.command_usage_counts) {
        std::cout << "  " << command << ": " << count << " times\n";
    }
    std::cout << "============================================\n";
}

bool PersonalVoiceCommands::save_config() {
    Json::Value root;
    Json::Value macros_json;

    {
        std::lock_guard<std::mutex> lock(macros_mutex_);
        for (const auto& [name, macro] : personal_macros_) {
            Json::Value macro_json;
            macro_json["name"] = macro->name;
            macro_json["enabled"] = macro->enabled;
            macro_json["privilege"] = static_cast<int>(macro->privilege);
            macro_json["execution_count"] = static_cast<Json::Int64>(macro->execution_count);

            Json::Value patterns(Json::arrayValue);
            for (const auto& pattern : macro->voice_patterns) {
                patterns.append(pattern);
            }
            macro_json["voice_patterns"] = patterns;

            Json::Value commands(Json::arrayValue);
            for (const auto& cmd : macro->commands) {
                commands.append(cmd);
            }
            macro_json["commands"] = commands;

            Json::Value apps(Json::arrayValue);
            for (const auto& app : macro->applications) {
                apps.append(app);
            }
            macro_json["applications"] = apps;

            Json::Value keys(Json::arrayValue);
            for (const auto& key : macro->keystrokes) {
                keys.append(key);
            }
            macro_json["keystrokes"] = keys;

            macros_json[name] = macro_json;
        }
    }

    root["personal_macros"] = macros_json;
    root["statistics"] = Json::Value();  // Add stats if needed

    // Expand tilde in config file path
    std::string config_path = config_.config_file;
    if (config_path[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            config_path = std::string(home) + config_path.substr(1);
        }
    }

    // Create directories if they don't exist
    size_t slash_pos = config_path.find_last_of('/');
    if (slash_pos != std::string::npos) {
        std::string dir = config_path.substr(0, slash_pos);
        system(("mkdir -p " + dir).c_str());
    }

    std::ofstream file(config_path);
    if (!file) {
        std::cerr << "[ERROR] Cannot save config to: " << config_path << "\n";
        return false;
    }

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &file);

    std::cout << "[INFO] Personal commands configuration saved to: " << config_path << "\n";
    return true;
}

bool PersonalVoiceCommands::load_config() {
    std::string config_path = config_.config_file;
    if (config_path[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            config_path = std::string(home) + config_path.substr(1);
        }
    }

    std::ifstream file(config_path);
    if (!file) {
        std::cout << "[INFO] No existing config file found at: " << config_path << "\n";
        return false;
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    if (!Json::parseFromStream(builder, file, &root, &errors)) {
        std::cerr << "[ERROR] Failed to parse config file: " << errors << "\n";
        return false;
    }

    if (root.isMember("personal_macros")) {
        return import_macros(root["personal_macros"]);
    }

    return true;
}

bool PersonalVoiceCommands::import_macros(const Json::Value& macros_json) {
    std::lock_guard<std::mutex> lock(macros_mutex_);

    for (const auto& name : macros_json.getMemberNames()) {
        const Json::Value& macro_json = macros_json[name];

        PersonalMacro macro(name);
        macro.enabled = macro_json.get("enabled", true).asBool();
        macro.privilege = static_cast<PrivilegeLevel>(macro_json.get("privilege", 0).asInt());
        macro.execution_count = macro_json.get("execution_count", 0).asUInt64();

        if (macro_json.isMember("voice_patterns")) {
            for (const auto& pattern : macro_json["voice_patterns"]) {
                macro.voice_patterns.push_back(pattern.asString());
            }
        }

        if (macro_json.isMember("commands")) {
            for (const auto& cmd : macro_json["commands"]) {
                macro.commands.push_back(cmd.asString());
            }
        }

        if (macro_json.isMember("applications")) {
            for (const auto& app : macro_json["applications"]) {
                macro.applications.push_back(app.asString());
            }
        }

        if (macro_json.isMember("keystrokes")) {
            for (const auto& key : macro_json["keystrokes"]) {
                macro.keystrokes.push_back(key.asString());
            }
        }

        personal_macros_[name] = std::make_unique<PersonalMacro>(macro);
    }

    std::cout << "[INFO] Loaded " << personal_macros_.size() << " personal macros from config\n";
    return true;
}

void PersonalVoiceCommands::cleanup_caches() {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto now = std::chrono::steady_clock::now();
    auto it = dtw_cache_.begin();

    while (it != dtw_cache_.end()) {
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.timestamp).count();
        if (age > config_.pattern_cache_ms) {
            it = dtw_cache_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace vtt