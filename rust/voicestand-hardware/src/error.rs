//! Hardware error types and result handling
//!
//! Comprehensive error handling for all hardware operations with zero unwrap() usage.

use std::fmt;
use thiserror::Error;

/// Hardware operation result type
pub type HardwareResult<T> = Result<T, HardwareError>;

/// Comprehensive hardware error types
#[derive(Error, Debug)]
pub enum HardwareError {
    /// Hardware subsystem not initialized
    #[error("Hardware subsystem not initialized")]
    NotInitialized,

    /// No hardware acceleration available
    #[error("No hardware acceleration available - NPU and GNA both failed")]
    NoHardwareAvailable,

    /// NPU device not available
    #[error("Intel NPU device not available")]
    NPUNotAvailable,

    /// GNA device not available
    #[error("Intel GNA device not available")]
    GNANotAvailable,

    /// Device is unhealthy
    #[error("Hardware device '{0}' is unhealthy")]
    DeviceUnhealthy(String),

    /// FFI operation failed
    #[error("FFI operation failed: {message}")]
    FFIError { message: String },

    /// Device initialization failed
    #[error("Device initialization failed: {device} - {reason}")]
    InitializationFailed { device: String, reason: String },

    /// Resource allocation failed
    #[error("Resource allocation failed: {resource} - {reason}")]
    ResourceAllocationFailed { resource: String, reason: String },

    /// Memory operation failed
    #[error("Memory operation failed: {operation} - {reason}")]
    MemoryError { operation: String, reason: String },

    /// Timeout occurred
    #[error("Operation timed out after {timeout_ms}ms: {operation}")]
    Timeout { operation: String, timeout_ms: u64 },

    /// Performance target not met
    #[error("Performance target not met: {metric} = {actual} (target: {target})")]
    PerformanceTarget {
        metric: String,
        actual: f64,
        target: f64,
    },

    /// Hardware driver error
    #[error("Hardware driver error: {driver} - {code}: {message}")]
    DriverError {
        driver: String,
        code: i32,
        message: String,
    },

    /// Model operation failed
    #[error("Model operation failed: {operation} - {reason}")]
    ModelError { operation: String, reason: String },

    /// Audio processing error
    #[error("Audio processing error: {stage} - {reason}")]
    AudioError { stage: String, reason: String },

    /// Configuration error
    #[error("Configuration error: {parameter} - {reason}")]
    ConfigError { parameter: String, reason: String },

    /// Power management error
    #[error("Power management error: {operation} - {reason}")]
    PowerError { operation: String, reason: String },

    /// Concurrency error
    #[error("Concurrency error: {operation} - {reason}")]
    ConcurrencyError { operation: String, reason: String },

    /// I/O operation failed
    #[error("I/O operation failed: {operation}")]
    IOError {
        operation: String,
        #[source]
        source: std::io::Error,
    },

    /// System error
    #[error("System error: {operation}")]
    SystemError {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl HardwareError {
    /// Create FFI error from message
    pub fn ffi_error(message: impl Into<String>) -> Self {
        Self::FFIError {
            message: message.into(),
        }
    }

    /// Create initialization error
    pub fn init_failed(device: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InitializationFailed {
            device: device.into(),
            reason: reason.into(),
        }
    }

    /// Create resource allocation error
    pub fn alloc_failed(resource: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ResourceAllocationFailed {
            resource: resource.into(),
            reason: reason.into(),
        }
    }

    /// Create memory error
    pub fn memory_error(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::MemoryError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create timeout error
    pub fn timeout(operation: impl Into<String>, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }

    /// Create performance error
    pub fn performance_target(
        metric: impl Into<String>,
        actual: f64,
        target: f64,
    ) -> Self {
        Self::PerformanceTarget {
            metric: metric.into(),
            actual,
            target,
        }
    }

    /// Create driver error
    pub fn driver_error(
        driver: impl Into<String>,
        code: i32,
        message: impl Into<String>,
    ) -> Self {
        Self::DriverError {
            driver: driver.into(),
            code,
            message: message.into(),
        }
    }

    /// Create model error
    pub fn model_error(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ModelError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create audio error
    pub fn audio_error(stage: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::AudioError {
            stage: stage.into(),
            reason: reason.into(),
        }
    }

    /// Create config error
    pub fn config_error(parameter: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigError {
            parameter: parameter.into(),
            reason: reason.into(),
        }
    }

    /// Create power error
    pub fn power_error(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::PowerError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create concurrency error
    pub fn concurrency_error(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConcurrencyError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create I/O error
    pub fn io_error(operation: impl Into<String>, source: std::io::Error) -> Self {
        Self::IOError {
            operation: operation.into(),
            source,
        }
    }

    /// Create system error
    pub fn system_error(
        operation: impl Into<String>,
        source: Box<dyn std::error::Error + Send + Sync>,
    ) -> Self {
        Self::SystemError {
            operation: operation.into(),
            source,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Non-recoverable errors
            Self::NoHardwareAvailable
            | Self::NPUNotAvailable
            | Self::GNANotAvailable
            | Self::NotInitialized => false,

            // Potentially recoverable errors
            Self::DeviceUnhealthy(_)
            | Self::Timeout { .. }
            | Self::PerformanceTarget { .. }
            | Self::AudioError { .. }
            | Self::ConcurrencyError { .. } => true,

            // Context-dependent errors
            Self::FFIError { .. }
            | Self::InitializationFailed { .. }
            | Self::ResourceAllocationFailed { .. }
            | Self::MemoryError { .. }
            | Self::DriverError { .. }
            | Self::ModelError { .. }
            | Self::ConfigError { .. }
            | Self::PowerError { .. }
            | Self::IOError { .. }
            | Self::SystemError { .. } => false,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::NoHardwareAvailable | Self::NotInitialized => ErrorSeverity::Critical,

            Self::NPUNotAvailable | Self::GNANotAvailable | Self::DeviceUnhealthy(_) => {
                ErrorSeverity::High
            }

            Self::PerformanceTarget { .. }
            | Self::Timeout { .. }
            | Self::DriverError { .. }
            | Self::ModelError { .. } => ErrorSeverity::Medium,

            Self::FFIError { .. }
            | Self::InitializationFailed { .. }
            | Self::ResourceAllocationFailed { .. }
            | Self::MemoryError { .. }
            | Self::AudioError { .. }
            | Self::ConfigError { .. }
            | Self::PowerError { .. }
            | Self::ConcurrencyError { .. }
            | Self::IOError { .. }
            | Self::SystemError { .. } => ErrorSeverity::Low,
        }
    }

    /// Get suggested recovery action
    pub fn recovery_suggestion(&self) -> Option<&'static str> {
        match self {
            Self::DeviceUnhealthy(_) => Some("Try device reset or reinitialize hardware"),
            Self::Timeout { .. } => Some("Retry operation with longer timeout"),
            Self::PerformanceTarget { .. } => Some("Check system load and CPU frequency"),
            Self::AudioError { .. } => Some("Check audio device configuration"),
            Self::ConcurrencyError { .. } => Some("Reduce concurrent operations"),
            Self::MemoryError { .. } => Some("Free memory and retry operation"),
            Self::PowerError { .. } => Some("Check power management settings"),
            _ => None,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue with degraded functionality
    Low,
    /// Medium severity - operation should be retried or alternative used
    Medium,
    /// High severity - major functionality impacted but system can continue
    High,
    /// Critical severity - system cannot function properly
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error context for additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub component: String,
    pub operation: String,
    pub timestamp: std::time::SystemTime,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(component: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            operation: operation.into(),
            timestamp: std::time::SystemTime::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }

    /// Add additional information to context
    pub fn with_info(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

/// Result extension for hardware operations
pub trait HardwareResultExt<T> {
    /// Add context to hardware error
    fn with_context(self, context: ErrorContext) -> HardwareResult<T>;

    /// Add simple context to hardware error
    fn with_simple_context(
        self,
        component: impl Into<String>,
        operation: impl Into<String>,
    ) -> HardwareResult<T>;

    /// Convert to HardwareError if not already
    fn into_hardware_error(self) -> HardwareResult<T>;
}

impl<T, E> HardwareResultExt<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context(self, _context: ErrorContext) -> HardwareResult<T> {
        self.map_err(|e| HardwareError::system_error("operation", Box::new(e)))
    }

    fn with_simple_context(
        self,
        component: impl Into<String>,
        operation: impl Into<String>,
    ) -> HardwareResult<T> {
        let context = ErrorContext::new(component, operation);
        self.with_context(context)
    }

    fn into_hardware_error(self) -> HardwareResult<T> {
        self.map_err(|e| HardwareError::system_error("unknown", Box::new(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = HardwareError::ffi_error("test message");
        assert!(matches!(error, HardwareError::FFIError { .. }));

        let error = HardwareError::init_failed("NPU", "driver not found");
        assert!(matches!(error, HardwareError::InitializationFailed { .. }));
    }

    #[test]
    fn test_error_severity() {
        let critical = HardwareError::NoHardwareAvailable;
        assert_eq!(critical.severity(), ErrorSeverity::Critical);

        let high = HardwareError::NPUNotAvailable;
        assert_eq!(high.severity(), ErrorSeverity::High);

        let medium = HardwareError::timeout("test", 1000);
        assert_eq!(medium.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_error_recoverability() {
        let non_recoverable = HardwareError::NoHardwareAvailable;
        assert!(!non_recoverable.is_recoverable());

        let recoverable = HardwareError::timeout("test", 1000);
        assert!(recoverable.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("NPU", "initialization")
            .with_info("device_id", "0")
            .with_info("driver_version", "1.0");

        assert_eq!(context.component, "NPU");
        assert_eq!(context.operation, "initialization");
        assert_eq!(context.additional_info.get("device_id"), Some(&"0".to_string()));
    }
}