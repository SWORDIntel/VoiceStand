#pragma once

#include "speaker_diarization.h"
#include "context_aware_processor.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <json/json.h>

namespace vtt {

// Meeting mode for multi-speaker transcription and management
class MeetingMode {
public:
    // Meeting participant
    struct Participant {
        std::string id;
        std::string name;
        std::string role;
        SpeakerEmbedding voice_profile;
        size_t word_count = 0;
        std::chrono::milliseconds total_speaking_time{0};
        std::chrono::steady_clock::time_point last_spoke;
        
        Participant(const std::string& participant_id, const std::string& participant_name)
            : id(participant_id), name(participant_name) {}
    };
    
    // Meeting transcript entry
    struct TranscriptEntry {
        std::string speaker_id;
        std::string speaker_name;
        std::string text;
        std::chrono::steady_clock::time_point timestamp;
        float confidence = 0.0f;
        bool is_action_item = false;
        bool is_decision = false;
        
        std::string to_string() const {
            auto time_t = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now() + 
                (timestamp - std::chrono::steady_clock::now()));
            
            std::stringstream ss;
            ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] ";
            ss << speaker_name << ": " << text;
            
            if (is_action_item) ss << " [ACTION]";
            if (is_decision) ss << " [DECISION]";
            
            return ss.str();
        }
    };
    
    // Meeting summary
    struct MeetingSummary {
        std::string title;
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
        std::vector<std::string> participants;
        std::vector<std::string> action_items;
        std::vector<std::string> decisions;
        std::vector<std::string> key_topics;
        std::unordered_map<std::string, size_t> speaker_word_counts;
        std::unordered_map<std::string, std::chrono::milliseconds> speaker_durations;
        
        Json::Value to_json() const {
            Json::Value summary;
            summary["title"] = title;
            
            // Convert time points to strings
            auto start_t = std::chrono::system_clock::to_time_t(start_time);
            auto end_t = std::chrono::system_clock::to_time_t(end_time);
            
            std::stringstream start_ss, end_ss;
            start_ss << std::put_time(std::localtime(&start_t), "%Y-%m-%d %H:%M:%S");
            end_ss << std::put_time(std::localtime(&end_t), "%Y-%m-%d %H:%M:%S");
            
            summary["start_time"] = start_ss.str();
            summary["end_time"] = end_ss.str();
            
            // Add participants
            Json::Value participants_json(Json::arrayValue);
            for (const auto& p : participants) {
                participants_json.append(p);
            }
            summary["participants"] = participants_json;
            
            // Add action items
            Json::Value actions_json(Json::arrayValue);
            for (const auto& action : action_items) {
                actions_json.append(action);
            }
            summary["action_items"] = actions_json;
            
            // Add decisions
            Json::Value decisions_json(Json::arrayValue);
            for (const auto& decision : decisions) {
                decisions_json.append(decision);
            }
            summary["decisions"] = decisions_json;
            
            // Add key topics
            Json::Value topics_json(Json::arrayValue);
            for (const auto& topic : key_topics) {
                topics_json.append(topic);
            }
            summary["key_topics"] = topics_json;
            
            // Add speaker statistics
            Json::Value speaker_stats;
            for (const auto& [speaker, count] : speaker_word_counts) {
                speaker_stats[speaker]["word_count"] = static_cast<Json::Int64>(count);
            }
            for (const auto& [speaker, duration] : speaker_durations) {
                speaker_stats[speaker]["speaking_time_seconds"] = 
                    static_cast<Json::Int64>(duration.count() / 1000);
            }
            summary["speaker_statistics"] = speaker_stats;
            
            return summary;
        }
    };
    
    // Meeting configuration
    struct Config {
        bool auto_identify_speakers = true;
        bool track_action_items = true;
        bool track_decisions = true;
        bool generate_summary = true;
        size_t max_participants = 10;
        size_t transcript_buffer_size = 1000;
        std::string output_format = "markdown";  // markdown, json, txt
        bool enable_real_time_captions = true;
    };
    
    MeetingMode(const Config& config = Config());
    ~MeetingMode();
    
    // Start a new meeting
    void start_meeting(const std::string& title = "");
    
    // End the current meeting
    MeetingSummary end_meeting();
    
    // Process audio with speaker identification
    void process_audio_segment(const float* samples, size_t num_samples,
                               uint32_t sample_rate, const std::string& transcription);
    
    // Add a participant
    void add_participant(const std::string& name, const std::string& role = "");
    
    // Register speaker voice profile
    void register_speaker_voice(const std::string& participant_id,
                                const float* voice_sample, size_t sample_size);
    
    // Get current transcript
    std::vector<TranscriptEntry> get_transcript() const { return transcript_; }
    
    // Get live caption for display
    std::string get_live_caption() const;
    
    // Export meeting transcript
    bool export_transcript(const std::string& filepath);
    
    // Export meeting summary
    bool export_summary(const std::string& filepath);
    
    // Mark last entry as action item
    void mark_as_action_item();
    
    // Mark last entry as decision
    void mark_as_decision();
    
    // Get meeting statistics
    struct Stats {
        size_t total_words = 0;
        size_t total_sentences = 0;
        size_t speaker_changes = 0;
        std::chrono::milliseconds total_duration{0};
        float average_confidence = 0.0f;
        size_t action_items_count = 0;
        size_t decisions_count = 0;
    };
    
    Stats get_stats() const;
    
    // Enable/disable features
    void set_auto_identify_speakers(bool enable) { config_.auto_identify_speakers = enable; }
    void set_track_action_items(bool enable) { config_.track_action_items = enable; }
    void set_track_decisions(bool enable) { config_.track_decisions = enable; }
    
private:
    // Identify speaker from audio
    std::string identify_speaker(const float* samples, size_t num_samples,
                                 uint32_t sample_rate);
    
    // Extract action items from text
    std::vector<std::string> extract_action_items(const std::string& text);
    
    // Extract decisions from text
    std::vector<std::string> extract_decisions(const std::string& text);
    
    // Extract key topics from transcript
    std::vector<std::string> extract_key_topics();
    
    // Update participant statistics
    void update_participant_stats(const std::string& participant_id,
                                  const std::string& text);
    
    // Generate markdown transcript
    std::string generate_markdown_transcript();
    
    // Generate JSON transcript
    std::string generate_json_transcript();
    
    // Generate plain text transcript
    std::string generate_text_transcript();
    
    Config config_;
    bool meeting_active_ = false;
    std::string meeting_title_;
    std::chrono::system_clock::time_point meeting_start_;
    std::chrono::system_clock::time_point meeting_end_;
    
    std::vector<std::shared_ptr<Participant>> participants_;
    std::vector<TranscriptEntry> transcript_;
    std::unique_ptr<SpeakerDiarization> speaker_diarization_;
    std::unique_ptr<ContextAwareProcessor> context_processor_;
    
    std::string current_speaker_id_;
    std::chrono::steady_clock::time_point last_speech_time_;
    
    // Action items and decisions
    std::vector<std::string> action_items_;
    std::vector<std::string> decisions_;
    
    Stats stats_;
};

// Implementation
inline MeetingMode::MeetingMode(const Config& config) 
    : config_(config) {
    speaker_diarization_ = std::make_unique<SpeakerDiarization>();
    context_processor_ = std::make_unique<ContextAwareProcessor>();
    
    // Set business context as default for meetings
    context_processor_->set_context_type(ContextAwareProcessor::ContextType::BUSINESS);
}

inline MeetingMode::~MeetingMode() {
    if (meeting_active_) {
        end_meeting();
    }
}

inline void MeetingMode::start_meeting(const std::string& title) {
    meeting_active_ = true;
    meeting_title_ = title.empty() ? "Meeting " + std::to_string(time(nullptr)) : title;
    meeting_start_ = std::chrono::system_clock::now();
    
    transcript_.clear();
    participants_.clear();
    action_items_.clear();
    decisions_.clear();
    stats_ = Stats();
    
    speaker_diarization_->clear();
    context_processor_->reset_context();
    
    std::cout << "[MEETING] Started: " << meeting_title_ << std::endl;
}

inline MeetingMode::MeetingSummary MeetingMode::end_meeting() {
    if (!meeting_active_) {
        return MeetingSummary();
    }
    
    meeting_active_ = false;
    meeting_end_ = std::chrono::system_clock::now();
    
    MeetingSummary summary;
    summary.title = meeting_title_;
    summary.start_time = meeting_start_;
    summary.end_time = meeting_end_;
    
    // Collect participant names
    for (const auto& participant : participants_) {
        summary.participants.push_back(participant->name);
        summary.speaker_word_counts[participant->name] = participant->word_count;
        summary.speaker_durations[participant->name] = participant->total_speaking_time;
    }
    
    // Copy action items and decisions
    summary.action_items = action_items_;
    summary.decisions = decisions_;
    
    // Extract key topics
    summary.key_topics = extract_key_topics();
    
    std::cout << "[MEETING] Ended: " << meeting_title_ << std::endl;
    std::cout << "  Duration: " 
              << std::chrono::duration_cast<std::chrono::minutes>(
                     meeting_end_ - meeting_start_).count() 
              << " minutes" << std::endl;
    std::cout << "  Participants: " << participants_.size() << std::endl;
    std::cout << "  Action Items: " << action_items_.size() << std::endl;
    std::cout << "  Decisions: " << decisions_.size() << std::endl;
    
    return summary;
}

inline void MeetingMode::process_audio_segment(const float* samples, size_t num_samples,
                                              uint32_t sample_rate, 
                                              const std::string& transcription) {
    if (!meeting_active_ || transcription.empty()) {
        return;
    }
    
    // Identify speaker
    std::string speaker_id = identify_speaker(samples, num_samples, sample_rate);
    
    // Process text with context awareness
    std::string processed_text = context_processor_->process_with_context(transcription);
    
    // Create transcript entry
    TranscriptEntry entry;
    entry.speaker_id = speaker_id;
    entry.timestamp = std::chrono::steady_clock::now();
    entry.text = processed_text;
    entry.confidence = 0.9f;  // Placeholder
    
    // Find speaker name
    auto participant_it = std::find_if(participants_.begin(), participants_.end(),
        [&speaker_id](const auto& p) { return p->id == speaker_id; });
    
    if (participant_it != participants_.end()) {
        entry.speaker_name = (*participant_it)->name;
    } else {
        entry.speaker_name = "Speaker " + speaker_id;
    }
    
    // Check for action items and decisions
    if (config_.track_action_items) {
        auto actions = extract_action_items(processed_text);
        if (!actions.empty()) {
            entry.is_action_item = true;
            for (const auto& action : actions) {
                action_items_.push_back(action);
            }
        }
    }
    
    if (config_.track_decisions) {
        auto decisions = extract_decisions(processed_text);
        if (!decisions.empty()) {
            entry.is_decision = true;
            for (const auto& decision : decisions) {
                decisions_.push_back(decision);
            }
        }
    }
    
    // Add to transcript
    transcript_.push_back(entry);
    
    // Limit transcript size
    if (transcript_.size() > config_.transcript_buffer_size) {
        transcript_.erase(transcript_.begin());
    }
    
    // Update speaker statistics
    update_participant_stats(speaker_id, processed_text);
    
    // Update general statistics
    stats_.total_words += std::count(processed_text.begin(), processed_text.end(), ' ') + 1;
    stats_.total_sentences++;
    
    if (speaker_id != current_speaker_id_) {
        stats_.speaker_changes++;
        current_speaker_id_ = speaker_id;
    }
    
    last_speech_time_ = std::chrono::steady_clock::now();
}

inline std::string MeetingMode::identify_speaker(const float* samples, size_t num_samples,
                                                uint32_t sample_rate) {
    if (!config_.auto_identify_speakers) {
        return "unknown";
    }
    
    // Use speaker diarization to identify speaker
    std::string speaker_id = speaker_diarization_->process_audio(samples, num_samples, sample_rate);
    
    // Try to match with registered participants
    for (const auto& participant : participants_) {
        // Simple matching - in production, use voice embedding comparison
        if (participant->id == speaker_id) {
            return participant->id;
        }
    }
    
    // Create new participant if not found
    if (participants_.size() < config_.max_participants) {
        auto new_participant = std::make_shared<Participant>(speaker_id, "Speaker " + speaker_id);
        participants_.push_back(new_participant);
    }
    
    return speaker_id;
}

inline void MeetingMode::add_participant(const std::string& name, const std::string& role) {
    if (participants_.size() >= config_.max_participants) {
        std::cerr << "[WARNING] Maximum number of participants reached" << std::endl;
        return;
    }
    
    std::string id = std::to_string(participants_.size() + 1);
    auto participant = std::make_shared<Participant>(id, name);
    participant->role = role;
    participants_.push_back(participant);
    
    std::cout << "[MEETING] Added participant: " << name;
    if (!role.empty()) {
        std::cout << " (" << role << ")";
    }
    std::cout << std::endl;
}

inline void MeetingMode::register_speaker_voice(const std::string& participant_id,
                                               const float* voice_sample, 
                                               size_t sample_size) {
    auto participant_it = std::find_if(participants_.begin(), participants_.end(),
        [&participant_id](const auto& p) { return p->id == participant_id; });
    
    if (participant_it != participants_.end()) {
        // Extract embedding from voice sample
        // Simplified - in production, use proper voice embedding extraction
        SpeakerEmbedding::Vector embedding(SpeakerEmbedding::EMBEDDING_DIM, 0.0f);
        for (size_t i = 0; i < std::min(sample_size, embedding.size()); ++i) {
            embedding[i] = voice_sample[i];
        }
        
        (*participant_it)->voice_profile = SpeakerEmbedding(embedding);
        (*participant_it)->voice_profile.normalize();
        
        std::cout << "[MEETING] Registered voice profile for: " 
                  << (*participant_it)->name << std::endl;
    }
}

inline std::vector<std::string> MeetingMode::extract_action_items(const std::string& text) {
    std::vector<std::string> actions;
    
    // Simple keyword-based extraction
    std::vector<std::string> action_keywords = {
        "action item", "todo", "task", "will do", "need to",
        "should", "must", "assign", "responsible for", "follow up"
    };
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& keyword : action_keywords) {
        if (lower_text.find(keyword) != std::string::npos) {
            actions.push_back(text);
            break;
        }
    }
    
    return actions;
}

inline std::vector<std::string> MeetingMode::extract_decisions(const std::string& text) {
    std::vector<std::string> decisions;
    
    // Simple keyword-based extraction
    std::vector<std::string> decision_keywords = {
        "decided", "decision", "agreed", "approved", "confirmed",
        "resolved", "concluded", "determined", "will proceed", "go ahead"
    };
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& keyword : decision_keywords) {
        if (lower_text.find(keyword) != std::string::npos) {
            decisions.push_back(text);
            break;
        }
    }
    
    return decisions;
}

inline std::vector<std::string> MeetingMode::extract_key_topics() {
    std::vector<std::string> topics;
    
    // Extract topics from context processor
    auto context_hint = context_processor_->get_context_hint();
    for (const auto& keyword : context_hint.keywords) {
        topics.push_back(keyword);
    }
    
    // Limit to top 10 topics
    if (topics.size() > 10) {
        topics.resize(10);
    }
    
    return topics;
}

inline void MeetingMode::update_participant_stats(const std::string& participant_id,
                                                 const std::string& text) {
    auto participant_it = std::find_if(participants_.begin(), participants_.end(),
        [&participant_id](const auto& p) { return p->id == participant_id; });
    
    if (participant_it != participants_.end()) {
        auto& participant = *participant_it;
        
        // Update word count
        size_t word_count = std::count(text.begin(), text.end(), ' ') + 1;
        participant->word_count += word_count;
        
        // Update speaking time
        auto now = std::chrono::steady_clock::now();
        if (participant->last_spoke.time_since_epoch().count() > 0) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - participant->last_spoke);
            
            // Only count if less than 10 seconds (to avoid long pauses)
            if (duration.count() < 10000) {
                participant->total_speaking_time += duration;
            }
        }
        participant->last_spoke = now;
    }
}

inline std::string MeetingMode::get_live_caption() const {
    if (transcript_.empty()) {
        return "";
    }
    
    // Get last 3 entries for context
    size_t start_idx = transcript_.size() > 3 ? transcript_.size() - 3 : 0;
    std::stringstream ss;
    
    for (size_t i = start_idx; i < transcript_.size(); ++i) {
        if (i > start_idx) ss << "\n";
        ss << transcript_[i].speaker_name << ": " << transcript_[i].text;
    }
    
    return ss.str();
}

inline bool MeetingMode::export_transcript(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    std::string content;
    
    if (config_.output_format == "markdown") {
        content = generate_markdown_transcript();
    } else if (config_.output_format == "json") {
        content = generate_json_transcript();
    } else {
        content = generate_text_transcript();
    }
    
    file << content;
    file.close();
    
    std::cout << "[MEETING] Transcript exported to: " << filepath << std::endl;
    return true;
}

inline std::string MeetingMode::generate_markdown_transcript() {
    std::stringstream ss;
    
    ss << "# " << meeting_title_ << "\n\n";
    
    auto start_t = std::chrono::system_clock::to_time_t(meeting_start_);
    ss << "**Date:** " << std::put_time(std::localtime(&start_t), "%Y-%m-%d") << "\n";
    ss << "**Time:** " << std::put_time(std::localtime(&start_t), "%H:%M:%S");
    
    if (meeting_end_.time_since_epoch().count() > 0) {
        auto end_t = std::chrono::system_clock::to_time_t(meeting_end_);
        ss << " - " << std::put_time(std::localtime(&end_t), "%H:%M:%S");
    }
    ss << "\n\n";
    
    // Participants
    ss << "## Participants\n\n";
    for (const auto& participant : participants_) {
        ss << "- " << participant->name;
        if (!participant->role.empty()) {
            ss << " (" << participant->role << ")";
        }
        ss << "\n";
    }
    ss << "\n";
    
    // Transcript
    ss << "## Transcript\n\n";
    for (const auto& entry : transcript_) {
        ss << "**" << entry.speaker_name << ":** " << entry.text;
        
        if (entry.is_action_item) {
            ss << " 📌";
        }
        if (entry.is_decision) {
            ss << " ✅";
        }
        
        ss << "\n\n";
    }
    
    // Action items
    if (!action_items_.empty()) {
        ss << "## Action Items\n\n";
        for (const auto& action : action_items_) {
            ss << "- [ ] " << action << "\n";
        }
        ss << "\n";
    }
    
    // Decisions
    if (!decisions_.empty()) {
        ss << "## Decisions\n\n";
        for (const auto& decision : decisions_) {
            ss << "- " << decision << "\n";
        }
        ss << "\n";
    }
    
    return ss.str();
}

inline MeetingMode::Stats MeetingMode::get_stats() const {
    Stats current_stats = stats_;
    
    if (meeting_active_) {
        auto now = std::chrono::steady_clock::now();
        auto start = std::chrono::steady_clock::time_point(meeting_start_.time_since_epoch());
        current_stats.total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start);
    }
    
    current_stats.action_items_count = action_items_.size();
    current_stats.decisions_count = decisions_.size();
    
    return current_stats;
}

}  // namespace vtt