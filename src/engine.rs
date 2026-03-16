use std::ffi::CString;
use std::ptr;

use crate::error::{Error, Result};
use crate::ffi;
use crate::session::{Session, SessionConfig};

/// Backend type for the inference engine.
pub enum Backend {
    Cpu,
    Gpu,
    Custom(String),
}

impl Backend {
    fn as_str(&self) -> &str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
            Backend::Custom(s) => s,
        }
    }
}

/// Builder for configuring and creating an [`Engine`].
pub struct EngineSettings {
    ptr: *mut ffi::LiteRtLmEngineSettings,
}

impl EngineSettings {
    /// Create new engine settings.
    ///
    /// - `model_path`: path to the model file on disk.
    /// - `backend`: compute backend to use.
    /// - `vision_backend`: optional vision backend.
    /// - `audio_backend`: optional audio backend.
    pub fn new(
        model_path: &str,
        backend: Backend,
        vision_backend: Option<&str>,
        audio_backend: Option<&str>,
    ) -> Result<Self> {
        let c_model_path = CString::new(model_path)?;
        let c_backend = CString::new(backend.as_str())?;
        let c_vision = vision_backend.map(CString::new).transpose()?;
        let c_audio = audio_backend.map(CString::new).transpose()?;

        let ptr = unsafe {
            ffi::litert_lm_engine_settings_create(
                c_model_path.as_ptr(),
                c_backend.as_ptr(),
                c_vision.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                c_audio.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            )
        };

        if ptr.is_null() {
            return Err(Error::EngineCreationFailed);
        }

        Ok(Self { ptr })
    }

    /// Set the maximum number of tokens the engine can handle.
    pub fn set_max_num_tokens(&mut self, max_num_tokens: i32) -> &mut Self {
        unsafe { ffi::litert_lm_engine_settings_set_max_num_tokens(self.ptr, max_num_tokens) };
        self
    }

    /// Set the cache directory for the engine.
    pub fn set_cache_dir(&mut self, cache_dir: &str) -> Result<&mut Self> {
        let c_dir = CString::new(cache_dir)?;
        unsafe { ffi::litert_lm_engine_settings_set_cache_dir(self.ptr, c_dir.as_ptr()) };
        Ok(self)
    }

    /// Set the activation data type (0=F32, 1=F16, 2=I16, 3=I8).
    pub fn set_activation_data_type(&mut self, dtype: i32) -> &mut Self {
        unsafe { ffi::litert_lm_engine_settings_set_activation_data_type(self.ptr, dtype) };
        self
    }

    /// Set the prefill chunk size (CPU backend with dynamic models only).
    pub fn set_prefill_chunk_size(&mut self, chunk_size: i32) -> &mut Self {
        unsafe { ffi::litert_lm_engine_settings_set_prefill_chunk_size(self.ptr, chunk_size) };
        self
    }

    /// Enable benchmarking.
    pub fn enable_benchmark(&mut self) -> &mut Self {
        unsafe { ffi::litert_lm_engine_settings_enable_benchmark(self.ptr) };
        self
    }

    /// Set the number of prefill tokens for benchmarking.
    pub fn set_num_prefill_tokens(&mut self, n: i32) -> &mut Self {
        unsafe { ffi::litert_lm_engine_settings_set_num_prefill_tokens(self.ptr, n) };
        self
    }

    /// Set the number of decode tokens for benchmarking.
    pub fn set_num_decode_tokens(&mut self, n: i32) -> &mut Self {
        unsafe { ffi::litert_lm_engine_settings_set_num_decode_tokens(self.ptr, n) };
        self
    }

    /// Build an [`Engine`] from these settings, consuming the settings.
    pub fn build(self) -> Result<Engine> {
        let engine_ptr = unsafe { ffi::litert_lm_engine_create(self.ptr) };
        if engine_ptr.is_null() {
            return Err(Error::EngineCreationFailed);
        }
        Ok(Engine { ptr: engine_ptr })
    }
}

impl Drop for EngineSettings {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_engine_settings_delete(self.ptr) };
        }
    }
}

// SAFETY: The underlying C++ object is self-contained.
unsafe impl Send for EngineSettings {}

/// A LiteRT-LM inference engine.
///
/// Created via [`EngineSettings::build`]. Use it to create [`Session`]s
/// or [`Conversation`](crate::conversation::Conversation)s.
pub struct Engine {
    pub(crate) ptr: *mut ffi::LiteRtLmEngine,
}

impl Engine {
    /// Create a new session from this engine with an optional config.
    pub fn create_session(&self, config: Option<&mut SessionConfig>) -> Result<Session> {
        let config_ptr = config.map_or(ptr::null_mut(), |c| c.ptr);
        let session_ptr = unsafe { ffi::litert_lm_engine_create_session(self.ptr, config_ptr) };
        if session_ptr.is_null() {
            return Err(Error::SessionCreationFailed);
        }
        Ok(Session { ptr: session_ptr })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_engine_delete(self.ptr) };
        }
    }
}

// SAFETY: The underlying C++ Engine is thread-safe for session creation.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}
