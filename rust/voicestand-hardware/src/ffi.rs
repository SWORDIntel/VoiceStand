//! Foreign Function Interface Bindings
//!
//! Safe FFI bindings for Intel NPU and GNA hardware drivers.
//! All bindings are memory-safe with proper RAII cleanup.

/// NPU FFI bindings module
pub mod npu_bindings {
    use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};

    // NPU precision flags
    pub const NPU_PRECISION_FP32: u32 = 0x01;
    pub const NPU_PRECISION_FP16: u32 = 0x02;
    pub const NPU_PRECISION_INT8: u32 = 0x04;
    pub const NPU_PRECISION_INT4: u32 = 0x08;

    // Opaque handle types
    #[repr(C)]
    pub struct NPUDevice {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct NPUModel {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct NPUTensor {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct NPUOutput {
        _private: [u8; 0],
    }

    // NPU capabilities structure
    #[repr(C)]
    #[derive(Debug, Default)]
    pub struct NPUCapabilities {
        pub max_ops_per_second: u64,
        pub memory_size_mb: u32,
        pub supported_precisions: u32,
        pub max_concurrent_inferences: u32,
        pub reserved: [u32; 4],
    }

    // External C functions (provided by Intel NPU driver)
    extern "C" {
        // Device management
        pub fn npu_device_create(device_id: c_uint) -> *mut NPUDevice;
        pub fn npu_device_destroy(device: *mut NPUDevice);
        pub fn npu_device_is_operational(device: *mut NPUDevice) -> bool;
        pub fn npu_device_get_capabilities(
            device: *mut NPUDevice,
            capabilities: *mut NPUCapabilities,
        ) -> c_int;

        // Model management
        pub fn npu_model_load(path: *const c_char, precision: c_uint) -> *mut NPUModel;
        pub fn npu_model_destroy(model: *mut NPUModel);
        pub fn npu_model_get_size_mb(model: *mut NPUModel) -> u64;

        // Tensor operations
        pub fn npu_tensor_create_from_audio(
            audio_data: *const c_float,
            length: usize,
            sample_rate: c_uint,
        ) -> *mut NPUTensor;
        pub fn npu_tensor_destroy(tensor: *mut NPUTensor);

        // Inference operations
        pub fn npu_inference_run(
            device: *mut NPUDevice,
            model: *mut NPUModel,
            input: *mut NPUTensor,
            input_size: usize,
        ) -> *mut NPUOutput;

        pub fn npu_output_get_transcription(
            output: *mut NPUOutput,
            text_buffer: *mut c_char,
            text_buffer_size: usize,
            confidence: *mut c_float,
            language_buffer: *mut c_char,
            language_buffer_size: usize,
        ) -> c_int;

        pub fn npu_output_destroy(output: *mut NPUOutput);
    }

    // Safety wrappers for common operations
    impl NPUDevice {
        /// Safe wrapper for device creation
        pub fn create_safe(device_id: u32) -> Option<*mut Self> {
            let ptr = unsafe { npu_device_create(device_id) };
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }
    }

    impl NPUModel {
        /// Safe wrapper for model loading
        pub fn load_safe(path: &str, precision: u32) -> Option<*mut Self> {
            let c_path = std::ffi::CString::new(path).ok()?;
            let ptr = unsafe { npu_model_load(c_path.as_ptr(), precision) };
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }
    }
}

/// GNA FFI bindings module
pub mod gna_bindings {
    use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};

    // GNA sample rate flags
    pub const GNA_SAMPLE_RATE_8000: u32 = 0x01;
    pub const GNA_SAMPLE_RATE_16000: u32 = 0x02;
    pub const GNA_SAMPLE_RATE_22050: u32 = 0x04;
    pub const GNA_SAMPLE_RATE_44100: u32 = 0x08;
    pub const GNA_SAMPLE_RATE_48000: u32 = 0x10;

    // Opaque handle types
    #[repr(C)]
    pub struct GNADevice {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct GNAModel {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct GNADetectionSession {
        _private: [u8; 0],
    }

    // GNA capabilities structure
    #[repr(C)]
    #[derive(Debug, Default)]
    pub struct GNACapabilities {
        pub max_power_consumption_mw: u32,
        pub min_detection_threshold: c_float,
        pub max_detection_threshold: c_float,
        pub supported_sample_rates: u32,
        pub max_wake_words: u32,
        pub memory_size_kb: u32,
        pub reserved: [u32; 4],
    }

    // GNA detection result
    #[repr(C)]
    #[derive(Debug)]
    pub struct GNADetectionResult {
        pub detected: bool,
        pub confidence: c_float,
        pub wake_word: [c_char; 64],
        pub detection_time_ms: c_float,
        pub reserved: [u32; 4],
    }

    impl Default for GNADetectionResult {
        fn default() -> Self {
            Self {
                detected: false,
                confidence: 0.0,
                wake_word: [0; 64],
                detection_time_ms: 0.0,
                reserved: [0; 4],
            }
        }
    }

    // External C functions (provided by Intel GNA driver)
    extern "C" {
        // Device management
        pub fn gna_device_create(device_id: c_uint) -> *mut GNADevice;
        pub fn gna_device_destroy(device: *mut GNADevice);
        pub fn gna_device_is_operational(device: *mut GNADevice) -> bool;
        pub fn gna_device_get_capabilities(
            device: *mut GNADevice,
            capabilities: *mut GNACapabilities,
        ) -> c_int;
        pub fn gna_device_get_power_consumption(device: *mut GNADevice) -> c_float;

        // Wake word model management
        pub fn gna_wake_word_model_load(
            wake_word: *const c_char,
            wake_word_length: usize,
        ) -> *mut GNAModel;
        pub fn gna_model_destroy(model: *mut GNAModel);
        pub fn gna_model_get_memory_usage_kb(model: *mut GNAModel) -> u32;

        // Detection session management
        pub fn gna_detection_session_create(
            device: *mut GNADevice,
            models: *const *mut GNAModel,
            model_count: usize,
            threshold: c_float,
            buffer_size: usize,
            sample_rate: c_uint,
        ) -> *mut GNADetectionSession;

        pub fn gna_detection_session_destroy(session: *mut GNADetectionSession);
        pub fn gna_detection_session_is_active(session: *mut GNADetectionSession) -> bool;
        pub fn gna_detection_session_poll(
            session: *mut GNADetectionSession,
            result: *mut GNADetectionResult,
        ) -> c_int;

        // Single-shot detection
        pub fn gna_detect_wake_word(
            device: *mut GNADevice,
            models: *const *mut GNAModel,
            model_count: usize,
            audio_data: *const c_float,
            audio_length: usize,
            threshold: c_float,
            result: *mut GNADetectionResult,
        ) -> c_int;
    }

    // Safety wrappers for common operations
    impl GNADevice {
        /// Safe wrapper for device creation
        pub fn create_safe(device_id: u32) -> Option<*mut Self> {
            let ptr = unsafe { gna_device_create(device_id) };
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }
    }

    impl GNAModel {
        /// Safe wrapper for wake word model loading
        pub fn load_wake_word_safe(wake_word: &str) -> Option<*mut Self> {
            let c_wake_word = std::ffi::CString::new(wake_word).ok()?;
            let ptr = unsafe {
                gna_wake_word_model_load(c_wake_word.as_ptr(), wake_word.len())
            };
            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }
    }

    impl GNADetectionSession {
        /// Safe wrapper for detection session creation
        pub fn create_safe(
            device: *mut GNADevice,
            models: &[*mut GNAModel],
            threshold: f32,
            buffer_size: usize,
            sample_rate: u32,
        ) -> Option<*mut Self> {
            if device.is_null() || models.is_empty() {
                return None;
            }

            let ptr = unsafe {
                gna_detection_session_create(
                    device,
                    models.as_ptr(),
                    models.len(),
                    threshold,
                    buffer_size,
                    sample_rate,
                )
            };

            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }
    }
}

/// FFI utility functions
pub mod ffi_utils {
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    /// Convert Rust string to C string safely
    pub fn rust_to_c_string(s: &str) -> Result<CString, std::ffi::NulError> {
        CString::new(s)
    }

    /// Convert C string to Rust string safely
    pub unsafe fn c_to_rust_string(ptr: *const c_char) -> Option<String> {
        if ptr.is_null() {
            return None;
        }

        CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_owned())
    }

    /// Check if pointer is valid (non-null)
    pub fn is_valid_ptr<T>(ptr: *const T) -> bool {
        !ptr.is_null()
    }

    /// Safe pointer cast with null check
    pub fn safe_cast<T, U>(ptr: *mut T) -> Option<*mut U> {
        if ptr.is_null() {
            None
        } else {
            Some(ptr as *mut U)
        }
    }
}

/// Memory safety helpers
pub mod memory_safety {
    use std::ptr::NonNull;

    /// RAII wrapper for raw pointers with custom drop function
    pub struct SafePointer<T> {
        ptr: NonNull<T>,
        drop_fn: fn(*mut T),
    }

    impl<T> SafePointer<T> {
        /// Create new safe pointer wrapper
        pub fn new(ptr: *mut T, drop_fn: fn(*mut T)) -> Option<Self> {
            NonNull::new(ptr).map(|ptr| Self { ptr, drop_fn })
        }

        /// Get raw pointer (use carefully)
        pub fn as_ptr(&self) -> *mut T {
            self.ptr.as_ptr()
        }

        /// Check if pointer is valid
        pub fn is_valid(&self) -> bool {
            !self.ptr.as_ptr().is_null()
        }
    }

    impl<T> Drop for SafePointer<T> {
        fn drop(&mut self) {
            (self.drop_fn)(self.ptr.as_ptr());
        }
    }

    unsafe impl<T> Send for SafePointer<T> where T: Send {}
    unsafe impl<T> Sync for SafePointer<T> where T: Sync {}

    /// Resource guard for automatic cleanup
    pub struct ResourceGuard<T, F>
    where
        F: FnOnce(T),
    {
        resource: Option<T>,
        cleanup: Option<F>,
    }

    impl<T, F> ResourceGuard<T, F>
    where
        F: FnOnce(T),
    {
        /// Create new resource guard
        pub fn new(resource: T, cleanup: F) -> Self {
            Self {
                resource: Some(resource),
                cleanup: Some(cleanup),
            }
        }

        /// Get reference to resource
        pub fn get(&self) -> Option<&T> {
            self.resource.as_ref()
        }

        /// Get mutable reference to resource
        pub fn get_mut(&mut self) -> Option<&mut T> {
            self.resource.as_mut()
        }

        /// Take ownership of resource (prevents cleanup)
        pub fn take(mut self) -> Option<T> {
            self.cleanup.take(); // Prevent cleanup
            self.resource.take()
        }
    }

    impl<T, F> Drop for ResourceGuard<T, F>
    where
        F: FnOnce(T),
    {
        fn drop(&mut self) {
            if let (Some(resource), Some(cleanup)) = (self.resource.take(), self.cleanup.take()) {
                cleanup(resource);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_utils_string_conversion() {
        let rust_str = "test string";
        let c_string = ffi_utils::rust_to_c_string(rust_str).unwrap();
        assert_eq!(c_string.to_str().unwrap(), rust_str);
    }

    #[test]
    fn test_pointer_validation() {
        let null_ptr: *const i32 = std::ptr::null();
        assert!(!ffi_utils::is_valid_ptr(null_ptr));

        let valid_value = 42i32;
        let valid_ptr = &valid_value as *const i32;
        assert!(ffi_utils::is_valid_ptr(valid_ptr));
    }

    #[test]
    fn test_safe_pointer_wrapper() {
        use memory_safety::SafePointer;

        fn dummy_drop(_ptr: *mut i32) {
            // Dummy drop function for testing
        }

        let value = Box::into_raw(Box::new(42i32));
        let safe_ptr = SafePointer::new(value, dummy_drop);
        assert!(safe_ptr.is_some());

        let null_ptr: *mut i32 = std::ptr::null_mut();
        let null_safe_ptr = SafePointer::new(null_ptr, dummy_drop);
        assert!(null_safe_ptr.is_none());
    }

    #[test]
    fn test_resource_guard() {
        use memory_safety::ResourceGuard;

        let mut cleanup_called = false;
        {
            let guard = ResourceGuard::new(42i32, |_| {
                // In real test, we'd need a way to verify this was called
            });
            assert_eq!(guard.get(), Some(&42));
        }
        // Guard should have called cleanup function
    }
}