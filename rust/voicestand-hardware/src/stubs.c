// Stub implementations for Intel NPU and GNA hardware functions
// These are placeholder implementations for development/testing without actual hardware

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// NPU stub implementations
typedef struct NPUDevice { int dummy; } NPUDevice;
typedef struct NPUModel { int dummy; } NPUModel;
typedef struct NPUTensor { int dummy; } NPUTensor;
typedef struct NPUOutput { int dummy; } NPUOutput;

NPUDevice* npu_device_create(uint32_t device_id) { return NULL; }
void npu_device_destroy(NPUDevice* device) {}
bool npu_device_is_operational(NPUDevice* device) { return false; }
int npu_device_get_capabilities(NPUDevice* device, void* capabilities) { return -1; }

NPUModel* npu_model_load(const char* path, uint32_t precision) { return NULL; }
void npu_model_destroy(NPUModel* model) {}
uint64_t npu_model_get_size_mb(NPUModel* model) { return 0; }

NPUTensor* npu_tensor_create_from_audio(const float* audio_data, size_t length, uint32_t sample_rate) { return NULL; }
void npu_tensor_destroy(NPUTensor* tensor) {}

NPUOutput* npu_inference_run(NPUDevice* device, NPUModel* model, NPUTensor* input, size_t input_size) { return NULL; }
int npu_output_get_transcription(NPUOutput* output, char* text_buffer, size_t text_buffer_size, float* confidence, char* language_buffer, size_t language_buffer_size) { return -1; }
void npu_output_destroy(NPUOutput* output) {}

// GNA stub implementations
typedef struct GNADevice { int dummy; } GNADevice;
typedef struct GNAModel { int dummy; } GNAModel;
typedef struct GNADetectionSession { int dummy; } GNADetectionSession;

GNADevice* gna_device_create(uint32_t device_id) { return NULL; }
void gna_device_destroy(GNADevice* device) {}
bool gna_device_is_operational(GNADevice* device) { return false; }
int gna_device_get_capabilities(GNADevice* device, void* capabilities) { return -1; }
float gna_device_get_power_consumption(GNADevice* device) { return 0.0f; }

GNAModel* gna_wake_word_model_load(const char* wake_word, size_t wake_word_length) { return NULL; }
void gna_model_destroy(GNAModel* model) {}
uint32_t gna_model_get_memory_usage_kb(GNAModel* model) { return 0; }

GNADetectionSession* gna_detection_session_create(GNADevice* device, GNAModel** models, size_t model_count, float threshold, size_t buffer_size, uint32_t sample_rate) { return NULL; }
void gna_detection_session_destroy(GNADetectionSession* session) {}
bool gna_detection_session_is_active(GNADetectionSession* session) { return false; }
int gna_detection_session_poll(GNADetectionSession* session, void* result) { return -1; }
int gna_detect_wake_word(GNADevice* device, GNAModel** models, size_t model_count, const float* audio_data, size_t audio_length, float threshold, void* result) { return -1; }
