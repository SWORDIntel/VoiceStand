# VoiceStand Learning System API Integration

This document describes the complete API integration layer for the VoiceStand C++ application to communicate with the learning system.

## Overview

The learning system integration provides:

- **HTTP Client Infrastructure**: Asynchronous HTTP communication with retry logic, offline queueing, and batch processing
- **Audio Feature Extraction**: Advanced MFCC, spectral, and prosodic feature extraction with JSON serialization
- **Ensemble Processing**: Multi-model recognition with adaptive weighting and UK English specialization
- **Adaptive Learning**: Pattern recognition and model optimization based on usage data
- **Complete Integration Layer**: High-level manager that coordinates all components

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Learning System Manager                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  API Client     │  │ Ensemble         │  │ Adaptive    │ │
│  │  - HTTP Client  │  │ Processor        │  │ Learning    │ │
│  │  - Async Queue  │  │ - Multi-model    │  │ - Patterns  │ │
│  │  - Offline      │  │ - UK Dialect     │  │ - Database  │ │
│  │  - Batching     │  │ - Weighting      │  │ - Analysis  │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ Feature         │  │ Integration      │                  │
│  │ Serializer      │  │ Layer            │                  │
│  │ - MFCC          │  │ - Callbacks      │                  │
│  │ - Spectral      │  │ - Coordination   │                  │
│  │ - JSON          │  │ - Error Handling │                  │
│  └─────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │   Learning System API   │
                  │   - Python Backend      │
                  │   - FastAPI             │
                  │   - Model Training      │
                  └─────────────────────────┘
```

## Core Components

### 1. LearningApiClient

**File**: `src/core/learning_api_client.h/cpp`

HTTP client with advanced features:

```cpp
// Configuration
LearningApiConfig config;
config.base_url = "http://localhost:5000";
config.api_key = "your-api-key";
config.enable_async_processing = true;
config.enable_offline_queue = true;
config.max_concurrent_requests = 5;

// Create and initialize client
LearningApiClient client(config);
client.initialize();

// Submit recognition results
RecognitionSubmission submission;
submission.text = "recognized text";
submission.confidence = 0.95;
submission.audio_features = {0.1, 0.2, 0.3};

client.submit_recognition_callback(submission, [](const HttpResponse& response) {
    if (response.success) {
        std::cout << "Submitted successfully" << std::endl;
    }
});
```

**Key Features**:
- Non-blocking HTTP requests to avoid audio latency
- Automatic retry logic with exponential backoff
- Offline request queueing when API is unavailable
- Batch processing for efficiency
- Thread-safe async processing

### 2. AudioFeatureSerializer

**File**: `src/core/audio_feature_serializer.h/cpp`

Advanced audio feature extraction:

```cpp
AudioFeatureSerializer serializer;

// Extract comprehensive features
auto features = serializer.extract_audio_features(samples, num_samples, sample_rate);

// Features include:
// - Energy, RMS, Zero-crossing rate
// - MFCC coefficients (13 by default)
// - Spectral centroid, rolloff, flux
// - Fundamental frequency and pitch variance
// - SNR and noise characteristics

// Serialize to JSON for API submission
Json::Value json_features = serializer.serialize_audio_features(features);

// Convert to ML feature vector
std::vector<double> feature_vector = serializer.audio_features_to_vector(features);
```

### 3. EnsembleWhisperProcessor

**File**: `src/core/ensemble_whisper_processor.h/cpp`

Multi-model recognition system:

```cpp
EnsembleWhisperProcessor::EnsembleConfig config;
config.model_paths = {"models/small.bin", "models/medium.bin", "models/large.bin"};
config.learning_mode = LearningMode::ADAPTIVE;
config.enable_uk_dialect_optimization = true;

EnsembleWhisperProcessor processor(config);
processor.initialize();

// Process audio with ensemble of models
auto result = processor.recognize(audio_data, num_samples, sample_rate);
std::cout << "Ensemble result: " << result.text
          << " (confidence: " << result.confidence
          << ", agreement: " << result.ensemble_agreement << ")" << std::endl;

// Report ground truth for learning
processor.report_ground_truth(result.text, "actual correct text");
```

### 4. AdaptiveLearningSystem

**File**: `src/core/adaptive_learning_system.h/cpp`

Pattern learning and model optimization:

```cpp
AdaptiveLearningSystem learning("postgresql://localhost/voicestand_learning");
learning.initialize();

// Record recognition contexts
RecognitionContext context;
context.text = "recognized text";
context.confidence = 0.88;
context.acoustic_features = {0.1, 0.2, 0.3};
context.speaker_id = "user123";

learning.record_recognition(context, "ground truth text");

// Analyze learned patterns
auto insights = learning.analyze_patterns();
for (const auto& insight : insights) {
    std::cout << "Learning insight: " << insight.description << std::endl;
}

// Get optimal model weights
auto weights = learning.get_optimal_model_weights();
```

### 5. LearningSystemManager

**File**: `src/core/learning_system_manager.h/cpp`

High-level coordinator:

```cpp
// Load configuration from settings
SettingsManager& settings = SettingsManager::instance();
auto learning_manager = create_learning_system_from_settings(settings);

// Integrate with existing whisper processor
WhisperProcessor whisper;
learning_manager->integrate_with_whisper_processor(&whisper);

// Submit manual corrections
learning_manager->submit_correction("recognized text", "correct text");

// Get system statistics
auto stats = learning_manager->get_learning_stats();
std::cout << "API online: " << stats.api_online << std::endl;
std::cout << "Pending requests: " << stats.pending_api_requests << std::endl;
```

## API Endpoints

The learning system expects these endpoints:

### POST /api/v1/recognition
Submit recognition results for analysis:
```json
{
  "text": "recognized text",
  "confidence": 0.95,
  "audio_features": [0.1, 0.2, 0.3, ...],
  "model_used": "whisper-base",
  "session_id": "session_123",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "metadata": {
    "is_ensemble": true,
    "ensemble_agreement": 0.87,
    "processing_time_ms": 250
  }
}
```

### GET /api/v1/models
Retrieve current model weights:
```json
[
  {
    "model_name": "whisper-base",
    "accuracy_score": 0.92,
    "weights": {
      "acoustic_weight": 1.2,
      "lexical_weight": 0.8
    },
    "version": "1.0.3",
    "updated_at": "2024-01-15T09:00:00.000Z"
  }
]
```

### POST /api/v1/training
Submit training corrections:
```json
{
  "recognized_text": "hello word",
  "corrected_text": "hello world",
  "audio_features": [0.1, 0.2, 0.3, ...],
  "confidence": 0.88,
  "correction_type": "manual_correction",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### GET /api/v1/feedback
Get accuracy feedback:
```json
{
  "recognition_id": "rec_123",
  "accuracy_score": 0.94,
  "feedback_type": "correct",
  "improvement_suggestions": [
    "Increase acoustic model weight",
    "Improve noise cancellation"
  ]
}
```

## Configuration

Add to your `~/.config/voice-to-text/settings.json`:

```json
{
  "learning_api": {
    "enabled": true,
    "base_url": "http://localhost:5000",
    "api_version": "v1",
    "api_key": "your-api-key-here",
    "connection_timeout_seconds": 10,
    "request_timeout_seconds": 30,
    "max_retries": 3,
    "enable_async_processing": true,
    "max_concurrent_requests": 5,
    "enable_batching": true,
    "batch_size": 10,
    "enable_offline_queue": true,
    "offline_queue_max_size": 10000
  },
  "ensemble": {
    "enabled": true,
    "learning_mode": "ADAPTIVE",
    "accuracy_target": 0.95,
    "confidence_threshold": 0.85,
    "min_ensemble_size": 3,
    "max_ensemble_size": 5,
    "enable_uk_dialect_optimization": true,
    "max_parallel_models": 3,
    "learning_update_interval_seconds": 300
  },
  "adaptive_learning": {
    "enabled": true,
    "database_url": "postgresql://localhost/voicestand_learning",
    "pattern_confidence_threshold": 0.7,
    "uk_pattern_bonus": 0.1,
    "max_history_size": 10000,
    "enable_uk_english_learning": true
  }
}
```

## Dependencies

Add to your system:

```bash
# Ubuntu/Debian
sudo apt install libcurl4-openssl-dev libfftw3-dev libpq-dev

# CentOS/RHEL
sudo yum install libcurl-devel fftw3-devel postgresql-devel

# Build system will automatically find these via pkg-config
```

CMakeLists.txt automatically includes:
- `libcurl` for HTTP client
- `fftw3` for audio feature extraction
- `jsoncpp` for JSON serialization
- `postgresql` libraries (if available)

## Usage Examples

### Basic Integration

```cpp
#include "core/learning_system_manager.h"

// Initialize with settings
auto learning_manager = create_learning_system_from_settings(settings);

// Integrate with whisper processor
learning_manager->integrate_with_whisper_processor(&whisper_processor);

// The system will automatically:
// 1. Extract audio features from processed audio
// 2. Submit recognition results to learning API
// 3. Apply received model weight updates
// 4. Learn from user corrections
```

### Manual Operations

```cpp
// Submit correction
learning_manager->submit_correction("hello word", "hello world");

// Request model updates
learning_manager->request_model_updates();

// Update speaker profile
std::vector<std::string> samples = {"Hello", "Goodbye", "Thank you"};
learning_manager->update_speaker_profile("user123", samples);

// Monitor statistics
auto stats = learning_manager->get_learning_stats();
```

### Advanced Ensemble Processing

```cpp
// Configure multiple models
EnsembleWhisperProcessor::EnsembleConfig config;
config.model_paths = {
    "models/ggml-small.bin",
    "models/ggml-medium.bin",
    "models/ggml-large.bin",
    "models/uk-english-fine-tuned.bin"
};
config.learning_mode = EnsembleWhisperProcessor::LearningMode::ADAPTIVE;
config.enable_uk_dialect_optimization = true;

auto ensemble = std::make_shared<EnsembleWhisperProcessor>(config);
ensemble->initialize();

// Process with ensemble
auto result = ensemble->recognize(audio_data, num_samples, sample_rate);

// Result includes ensemble-specific information
std::cout << "Ensemble agreement: " << result.ensemble_agreement << std::endl;
for (const auto& model_conf : result.model_confidences) {
    std::cout << model_conf.first << ": " << model_conf.second << std::endl;
}
```

## Error Handling

The system includes comprehensive error handling:

- **Network failures**: Automatic retry with exponential backoff
- **API unavailable**: Offline request queueing with persistence
- **Model failures**: Graceful fallback to available models
- **Database errors**: Continue operation without persistence
- **Configuration errors**: Use sensible defaults with warnings

## Performance Considerations

- **Non-blocking operations**: All API calls are asynchronous to avoid audio latency
- **Efficient batching**: Multiple requests are batched to reduce API overhead
- **Memory management**: Uses memory pools and circular buffers for audio processing
- **Thread safety**: All components are thread-safe with proper synchronization
- **Resource limits**: Configurable limits on queue sizes and concurrent requests

## Monitoring and Statistics

The system provides comprehensive monitoring:

```cpp
auto stats = learning_manager->get_learning_stats();
std::cout << "API Status: " << (stats.api_online ? "Online" : "Offline") << std::endl;
std::cout << "Recognitions Submitted: " << stats.total_recognitions_submitted << std::endl;
std::cout << "Corrections Submitted: " << stats.total_corrections_submitted << std::endl;
std::cout << "Pending Requests: " << stats.pending_api_requests << std::endl;
std::cout << "Offline Queue Size: " << stats.offline_requests_queued << std::endl;
std::cout << "Current Accuracy: " << stats.current_ensemble_accuracy << std::endl;
```

## Integration with Existing Code

The learning system integrates seamlessly with existing VoiceStand components:

1. **WhisperProcessor**: Automatically submits recognition results
2. **Settings Manager**: Loads configuration from settings file
3. **Audio Capture**: Extracts features from audio stream
4. **GUI**: Can display learning statistics and status
5. **Hotkey Manager**: Can trigger manual learning operations

## Best Practices

1. **Gradual Rollout**: Start with learning disabled, then enable components incrementally
2. **Monitor Performance**: Watch for audio latency increases with learning enabled
3. **API Key Security**: Store API keys securely, not in source code
4. **Database Maintenance**: Regularly clean old learning patterns from database
5. **Model Updates**: Test model updates in development before production deployment
6. **User Privacy**: Ensure user consent for data collection and learning

## Troubleshooting

### Common Issues

1. **API Connection Failures**:
   - Check network connectivity
   - Verify API endpoint URL and key
   - Review firewall settings

2. **Audio Latency**:
   - Reduce `max_concurrent_requests`
   - Disable synchronous operations
   - Increase batch timeout

3. **Memory Usage**:
   - Reduce `max_history_size` in adaptive learning
   - Lower `offline_queue_max_size`
   - Use smaller audio feature vectors

4. **Database Errors**:
   - Verify PostgreSQL connection string
   - Check database permissions
   - Ensure required extensions (pgvector) are installed

### Debug Mode

Enable detailed logging by setting environment variables:

```bash
export VOICESTAND_DEBUG=1
export VOICESTAND_LOG_LEVEL=DEBUG
./voice-to-text
```

This will provide detailed information about:
- API request/response cycles
- Learning pattern analysis
- Model weight updates
- Feature extraction process
- Component initialization status

## Future Enhancements

Planned improvements for the learning system:

1. **Real-time Model Adaptation**: Apply weight updates without restart
2. **Privacy-Preserving Learning**: Federated learning with differential privacy
3. **Advanced Feature Engineering**: Deep learning feature extraction
4. **Multi-Language Support**: Specialized learning for multiple languages
5. **Edge Computing**: Local learning without API dependency
6. **Voice Biometrics**: Speaker identification and adaptation
7. **Context Awareness**: Meeting detection and domain adaptation

For the latest updates and detailed API documentation, visit the [VoiceStand Learning System Documentation](https://github.com/SWORDIntel/VoiceStand/tree/main/docs/learning-system).