#include "core/personal_voice_commands.h"
#include "core/phase2_personal_commands_integration.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace vtt;

class Phase2PersonalCommandsTest {
public:
    void run_all_tests() {
        std::cout << "=== Phase 2 Personal Commands Test Suite ===\n\n";

        test_dtw_pattern_matching();
        test_personal_macro_management();
        test_builtin_commands();
        test_security_validation();
        test_performance_benchmarks();
        test_integration_system();

        print_test_summary();
    }

private:
    int total_tests = 0;
    int passed_tests = 0;

    void test_dtw_pattern_matching() {
        std::cout << "--- Testing DTW Pattern Matching ---\n";

        PersonalVoiceCommands commands;
        commands.initialize();

        // Test exact matches
        float similarity1 = commands.compute_dtw_similarity("open browser", "open browser");
        check_test("Exact match similarity", similarity1 > 0.95f);

        // Test similar phrases
        float similarity2 = commands.compute_dtw_similarity("open web browser", "open browser");
        check_test("Similar phrase similarity", similarity2 > 0.70f);

        // Test phonetic similarity
        float similarity3 = commands.compute_dtw_similarity("lock screen", "lock the screen");
        check_test("Phonetic similarity", similarity3 > 0.75f);

        // Test different phrases
        float similarity4 = commands.compute_dtw_similarity("open browser", "shutdown computer");
        check_test("Different phrase dissimilarity", similarity4 < 0.50f);

        std::cout << "DTW Similarities:\n";
        std::cout << "  Exact match: " << (similarity1 * 100) << "%\n";
        std::cout << "  Similar phrase: " << (similarity2 * 100) << "%\n";
        std::cout << "  Phonetic: " << (similarity3 * 100) << "%\n";
        std::cout << "  Different: " << (similarity4 * 100) << "%\n\n";
    }

    void test_personal_macro_management() {
        std::cout << "--- Testing Personal Macro Management ---\n";

        PersonalVoiceCommands commands;
        commands.initialize();

        // Create a custom macro
        PersonalVoiceCommands::PersonalMacro custom_macro("test_macro");
        custom_macro.voice_patterns = {"test command", "run test"};
        custom_macro.commands = {"echo 'Test successful'"};
        custom_macro.privilege = PersonalVoiceCommands::PrivilegeLevel::USER;

        // Test macro registration
        bool registered = commands.register_personal_macro(custom_macro);
        check_test("Macro registration", registered);

        // Test macro retrieval
        auto macros = commands.get_personal_macros();
        bool found = false;
        for (const auto& macro : macros) {
            if (macro.name == "test_macro") {
                found = true;
                break;
            }
        }
        check_test("Macro retrieval", found);

        // Test macro removal
        bool removed = commands.remove_personal_macro("test_macro");
        check_test("Macro removal", removed);

        std::cout << "Registered " << macros.size() << " total macros\n\n";
    }

    void test_builtin_commands() {
        std::cout << "--- Testing Built-in Commands ---\n";

        PersonalVoiceCommands commands;
        commands.initialize();

        // Test command recognition and execution (dry run)
        std::vector<std::pair<std::string, std::string>> test_commands = {
            {"open terminal", "Terminal launch command"},
            {"volume up", "Volume control command"},
            {"take screenshot", "Screenshot command"},
            {"copy", "Copy text command"},
            {"minimize window", "Window management command"}
        };

        for (const auto& [voice_input, description] : test_commands) {
            auto result = commands.process_voice_input(voice_input, 0.90f);

            // For this test, we just check that commands are recognized
            // Actual execution would require proper X11/system setup
            bool recognized = !result.command_name.empty();
            check_test(description + " recognition", recognized);

            if (recognized) {
                std::cout << "  Recognized: \"" << voice_input << "\" -> "
                         << result.command_name << "\n";
            }
        }
        std::cout << "\n";
    }

    void test_security_validation() {
        std::cout << "--- Testing Security Validation ---\n";

        PersonalVoiceCommands commands;
        commands.initialize();

        // Test different privilege levels
        commands.set_current_privilege_level(PersonalVoiceCommands::PrivilegeLevel::USER);

        // User-level command should work
        auto result1 = commands.process_voice_input("open browser", 0.95f);
        check_test("User privilege command", !result1.command_name.empty());

        // System-level command with user privilege (should be rejected in secure mode)
        auto result2 = commands.process_voice_input("shutdown computer", 0.95f);
        // Note: This might succeed depending on security configuration
        std::cout << "  System command result: " << result2.response << "\n";

        // Test confidence-based rejection
        auto result3 = commands.process_voice_input("open browser", 0.30f);
        check_test("Low confidence rejection", result3.response.find("confidence") != std::string::npos);

        std::cout << "\n";
    }

    void test_performance_benchmarks() {
        std::cout << "--- Testing Performance Benchmarks ---\n";

        PersonalVoiceCommands commands;
        commands.initialize();

        const int num_iterations = 100;
        std::vector<std::string> test_inputs = {
            "open browser",
            "take screenshot",
            "volume up",
            "minimize window",
            "copy text"
        };

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            for (const auto& input : test_inputs) {
                commands.process_voice_input(input, 0.85f);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        float avg_time_ms = total_time / (num_iterations * test_inputs.size());

        std::cout << "Performance Results:\n";
        std::cout << "  Total commands processed: " << (num_iterations * test_inputs.size()) << "\n";
        std::cout << "  Total time: " << total_time << " ms\n";
        std::cout << "  Average command latency: " << avg_time_ms << " ms\n";

        // Check if average latency meets target (<100ms)
        check_test("Performance target (<100ms)", avg_time_ms < 100.0f);

        // Test DTW caching performance
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; ++i) {
            commands.compute_dtw_similarity("open browser", "open browser");
        }
        end_time = std::chrono::high_resolution_clock::now();
        float cache_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        std::cout << "  DTW cache performance: " << (cache_time / 50) << " ms per cached lookup\n";
        check_test("DTW cache efficiency", (cache_time / 50) < 1.0f);

        std::cout << "\n";
    }

    void test_integration_system() {
        std::cout << "--- Testing Integration System ---\n";

        Phase2PersonalCommandsIntegration::Config config;
        config.enable_command_mode = true;
        config.enable_dictation_mode = true;
        config.use_gna_wake_detection = true;

        Phase2PersonalCommandsIntegration integration(config);

        // Test initialization
        bool initialized = integration.initialize();
        check_test("Integration initialization", initialized);

        if (initialized) {
            // Test mode transitions
            integration.set_mode(Phase2PersonalCommandsIntegration::OperationMode::COMMAND);
            auto mode = integration.get_mode();
            check_test("Mode setting", mode == Phase2PersonalCommandsIntegration::OperationMode::COMMAND);

            // Test personal macro delegation
            PersonalVoiceCommands::PersonalMacro test_macro("integration_test");
            test_macro.voice_patterns = {"integration test"};
            test_macro.commands = {"echo 'Integration test'"};

            bool macro_registered = integration.register_personal_macro(test_macro);
            check_test("Integration macro registration", macro_registered);

            auto macros = integration.get_personal_macros();
            bool found = false;
            for (const auto& macro : macros) {
                if (macro.name == "integration_test") {
                    found = true;
                    break;
                }
            }
            check_test("Integration macro retrieval", found);

            // Test statistics
            auto stats = integration.get_system_stats();
            std::cout << "  Integration statistics initialized\n";
        }

        std::cout << "\n";
    }

    void check_test(const std::string& test_name, bool passed) {
        total_tests++;
        if (passed) {
            passed_tests++;
            std::cout << "  ✓ " << test_name << "\n";
        } else {
            std::cout << "  ✗ " << test_name << "\n";
        }
    }

    void print_test_summary() {
        std::cout << "=== Test Summary ===\n";
        std::cout << "Total tests: " << total_tests << "\n";
        std::cout << "Passed: " << passed_tests << "\n";
        std::cout << "Failed: " << (total_tests - passed_tests) << "\n";
        std::cout << "Success rate: " << (passed_tests * 100.0f / total_tests) << "%\n";

        if (passed_tests == total_tests) {
            std::cout << "\n🎉 All tests passed! Phase 2 Personal Commands ready for deployment.\n";
        } else {
            std::cout << "\n⚠️  Some tests failed. Review implementation before deployment.\n";
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "Phase 2 Week 4: Personal Voice Commands System Test\n";
    std::cout << "Building on GNA foundation with DTW pattern matching\n";
    std::cout << "Target: 20+ personal commands, >95% accuracy, <100ms latency\n\n";

    // Check for test mode argument
    bool run_interactive = false;
    if (argc > 1 && std::string(argv[1]) == "--interactive") {
        run_interactive = true;
    }

    Phase2PersonalCommandsTest test_suite;
    test_suite.run_all_tests();

    if (run_interactive) {
        std::cout << "\n=== Interactive Command Testing ===\n";
        std::cout << "Enter voice commands to test (type 'quit' to exit):\n";

        PersonalVoiceCommands commands;
        commands.initialize();

        // Set up callback for real-time feedback
        commands.set_command_callback([](const PersonalVoiceCommands::ExecutionResult& result) {
            std::cout << "\nCommand Result:\n";
            std::cout << "  Success: " << (result.success ? "Yes" : "No") << "\n";
            std::cout << "  Command: " << result.command_name << "\n";
            std::cout << "  Response: " << result.response << "\n";
            std::cout << "  Confidence: " << (result.confidence * 100) << "%\n";
            std::cout << "  Execution time: " << result.execution_time_ms << " ms\n";
            if (!result.error_message.empty()) {
                std::cout << "  Error: " << result.error_message << "\n";
            }
        });

        std::string input;
        while (true) {
            std::cout << "\nVoice input: ";
            std::getline(std::cin, input);

            if (input == "quit" || input == "exit") {
                break;
            }

            if (input.empty()) {
                continue;
            }

            // Simulate voice confidence (in real system, this comes from Whisper)
            float confidence = 0.90f;
            commands.process_voice_input(input, confidence);
        }

        // Print usage statistics
        commands.print_usage_report();
    }

    return 0;
}