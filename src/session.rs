use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::ptr;

use crate::error::{Error, Result};
use crate::ffi;
use crate::StreamCallbackBox;

/// Sampler type for token selection.
#[derive(Debug, Clone, Copy, Default)]
pub enum SamplerType {
    #[default]
    Unspecified,
    TopK,
    TopP,
    Greedy,
}

impl SamplerType {
    fn to_ffi(self) -> ffi::Type {
        match self {
            SamplerType::Unspecified => ffi::Type_kTypeUnspecified,
            SamplerType::TopK => ffi::Type_kTopK,
            SamplerType::TopP => ffi::Type_kTopP,
            SamplerType::Greedy => ffi::Type_kGreedy,
        }
    }
}

/// Parameters for the token sampler.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub sampler_type: SamplerType,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: i32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            sampler_type: SamplerType::Unspecified,
            top_k: 40,
            top_p: 0.95,
            temperature: 1.0,
            seed: 0,
        }
    }
}

impl SamplerParams {
    fn to_ffi(&self) -> ffi::LiteRtLmSamplerParams {
        ffi::LiteRtLmSamplerParams {
            type_: self.sampler_type.to_ffi(),
            top_k: self.top_k,
            top_p: self.top_p,
            temperature: self.temperature,
            seed: self.seed,
        }
    }
}

/// Configuration for a session.
pub struct SessionConfig {
    pub(crate) ptr: *mut ffi::LiteRtLmSessionConfig,
}

impl SessionConfig {
    /// Create a new default session config.
    pub fn new() -> Result<Self> {
        let ptr = unsafe { ffi::litert_lm_session_config_create() };
        if ptr.is_null() {
            return Err(Error::SessionConfigCreationFailed);
        }
        Ok(Self { ptr })
    }

    /// Set the maximum number of output tokens per decode step.
    pub fn set_max_output_tokens(&mut self, max_output_tokens: i32) -> &mut Self {
        unsafe { ffi::litert_lm_session_config_set_max_output_tokens(self.ptr, max_output_tokens) };
        self
    }

    /// Set the sampler parameters.
    pub fn set_sampler_params(&mut self, params: &SamplerParams) -> &mut Self {
        let ffi_params = params.to_ffi();
        unsafe { ffi::litert_lm_session_config_set_sampler_params(self.ptr, &ffi_params) };
        self
    }
}

impl Drop for SessionConfig {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_session_config_delete(self.ptr) };
        }
    }
}

unsafe impl Send for SessionConfig {}

/// Type of input data for multimodal content generation.
pub enum InputKind<'a> {
    /// UTF-8 text input.
    Text(&'a str),
    /// Raw image bytes.
    Image(&'a [u8]),
    /// Marker for end of image input.
    ImageEnd,
    /// Raw audio bytes.
    Audio(&'a [u8]),
    /// Marker for end of audio input.
    AudioEnd,
}

/// Build an array of FFI `InputData` from a slice of `InputKind`.
fn build_input_data(inputs: &[InputKind<'_>]) -> Result<Vec<ffi::InputData>> {
    let mut c_strings = Vec::new();
    let mut ffi_inputs = Vec::with_capacity(inputs.len());

    for input in inputs {
        match input {
            InputKind::Text(text) => {
                let c_str = CString::new(*text)?;
                let data = ffi::InputData {
                    type_: ffi::InputDataType_kInputText,
                    data: c_str.as_ptr() as *const c_void,
                    size: c_str.as_bytes().len(),
                };
                ffi_inputs.push(data);
                c_strings.push(c_str);
            }
            InputKind::Image(bytes) => {
                ffi_inputs.push(ffi::InputData {
                    type_: ffi::InputDataType_kInputImage,
                    data: bytes.as_ptr() as *const c_void,
                    size: bytes.len(),
                });
            }
            InputKind::ImageEnd => {
                ffi_inputs.push(ffi::InputData {
                    type_: ffi::InputDataType_kInputImageEnd,
                    data: ptr::null(),
                    size: 0,
                });
            }
            InputKind::Audio(bytes) => {
                ffi_inputs.push(ffi::InputData {
                    type_: ffi::InputDataType_kInputAudio,
                    data: bytes.as_ptr() as *const c_void,
                    size: bytes.len(),
                });
            }
            InputKind::AudioEnd => {
                ffi_inputs.push(ffi::InputData {
                    type_: ffi::InputDataType_kInputAudioEnd,
                    data: ptr::null(),
                    size: 0,
                });
            }
        }
    }

    // Keep c_strings alive — they are referenced by ffi_inputs pointers.
    std::mem::forget(c_strings);
    Ok(ffi_inputs)
}

/// A LiteRT-LM inference session.
///
/// Created via [`Engine::create_session`](crate::engine::Engine::create_session).
pub struct Session {
    pub(crate) ptr: *mut ffi::LiteRtLmSession,
}

impl Session {
    /// Generate content from multimodal inputs (blocking).
    ///
    /// Returns the generated text responses.
    pub fn generate_content(&self, inputs: &[InputKind<'_>]) -> Result<Responses> {
        let ffi_inputs = build_input_data(inputs)?;
        let responses_ptr = unsafe {
            ffi::litert_lm_session_generate_content(self.ptr, ffi_inputs.as_ptr(), ffi_inputs.len())
        };
        if responses_ptr.is_null() {
            return Err(Error::GenerationFailed);
        }
        Ok(Responses { ptr: responses_ptr })
    }

    /// Generate content with streaming via a callback (non-blocking).
    ///
    /// The callback receives `(chunk, is_final, error_message)`.
    pub fn generate_content_stream<F>(&self, inputs: &[InputKind<'_>], callback: F) -> Result<()>
    where
        F: FnMut(&str, bool, Option<&str>) + Send + 'static,
    {
        let ffi_inputs = build_input_data(inputs)?;
        let callback_box: Box<StreamCallbackBox> = Box::new(Box::new(callback));
        let callback_data = Box::into_raw(callback_box) as *mut c_void;

        let ret = unsafe {
            ffi::litert_lm_session_generate_content_stream(
                self.ptr,
                ffi_inputs.as_ptr(),
                ffi_inputs.len(),
                Some(stream_callback_trampoline),
                callback_data,
            )
        };

        if ret != 0 {
            // Reclaim the callback to avoid leak on error.
            unsafe {
                drop(Box::from_raw(callback_data as *mut StreamCallbackBox));
            }
            return Err(Error::StreamingFailed(ret));
        }
        Ok(())
    }

    /// Get benchmark info from the session.
    pub fn get_benchmark_info(&self) -> Result<BenchmarkInfo> {
        let ptr = unsafe { ffi::litert_lm_session_get_benchmark_info(self.ptr) };
        if ptr.is_null() {
            return Err(Error::BenchmarkInfoFailed);
        }
        Ok(BenchmarkInfo { ptr })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_session_delete(self.ptr) };
        }
    }
}

unsafe impl Send for Session {}

/// C-compatible trampoline for the streaming callback.
unsafe extern "C" fn stream_callback_trampoline(
    callback_data: *mut c_void,
    chunk: *const std::os::raw::c_char,
    is_final: bool,
    error_msg: *const std::os::raw::c_char,
) {
    let cb = &mut *(callback_data as *mut StreamCallbackBox);

    let chunk_str = if chunk.is_null() {
        ""
    } else {
        CStr::from_ptr(chunk).to_str().unwrap_or("")
    };

    let error_str = if error_msg.is_null() {
        None
    } else {
        Some(
            CStr::from_ptr(error_msg)
                .to_str()
                .unwrap_or("unknown error"),
        )
    };

    cb(chunk_str, is_final, error_str);

    // If this is the final call, reclaim the callback box.
    if is_final {
        drop(Box::from_raw(
            callback_data as *mut Box<dyn FnMut(&str, bool, Option<&str>) + Send>,
        ));
    }
}

/// Collection of generated responses from a session.
pub struct Responses {
    ptr: *mut ffi::LiteRtLmResponses,
}

impl Responses {
    /// Number of response candidates.
    pub fn num_candidates(&self) -> i32 {
        unsafe { ffi::litert_lm_responses_get_num_candidates(self.ptr) }
    }

    /// Get the response text at the given index.
    pub fn text_at(&self, index: i32) -> Result<&str> {
        let count = self.num_candidates();
        if index < 0 || index >= count {
            return Err(Error::IndexOutOfBounds { index, count });
        }
        let ptr = unsafe { ffi::litert_lm_responses_get_response_text_at(self.ptr, index) };
        if ptr.is_null() {
            return Err(Error::IndexOutOfBounds { index, count });
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        Ok(cstr.to_str().unwrap_or(""))
    }

    /// Convenience: get the first response text.
    pub fn first_text(&self) -> Result<&str> {
        self.text_at(0)
    }
}

impl Drop for Responses {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_responses_delete(self.ptr) };
        }
    }
}

unsafe impl Send for Responses {}

/// Benchmark information from a session or conversation.
pub struct BenchmarkInfo {
    pub(crate) ptr: *mut ffi::LiteRtLmBenchmarkInfo,
}

impl BenchmarkInfo {
    /// Time to the first token in seconds.
    pub fn time_to_first_token(&self) -> f64 {
        unsafe { ffi::litert_lm_benchmark_info_get_time_to_first_token(self.ptr) }
    }

    /// Total initialization time in seconds.
    pub fn total_init_time(&self) -> f64 {
        unsafe { ffi::litert_lm_benchmark_info_get_total_init_time_in_second(self.ptr) }
    }

    /// Number of prefill turns.
    pub fn num_prefill_turns(&self) -> i32 {
        unsafe { ffi::litert_lm_benchmark_info_get_num_prefill_turns(self.ptr) }
    }

    /// Number of decode turns.
    pub fn num_decode_turns(&self) -> i32 {
        unsafe { ffi::litert_lm_benchmark_info_get_num_decode_turns(self.ptr) }
    }

    /// Prefill token count at a given turn.
    pub fn prefill_token_count_at(&self, index: i32) -> i32 {
        unsafe { ffi::litert_lm_benchmark_info_get_prefill_token_count_at(self.ptr, index) }
    }

    /// Decode token count at a given turn.
    pub fn decode_token_count_at(&self, index: i32) -> i32 {
        unsafe { ffi::litert_lm_benchmark_info_get_decode_token_count_at(self.ptr, index) }
    }

    /// Prefill tokens per second at a given turn.
    pub fn prefill_tokens_per_sec_at(&self, index: i32) -> f64 {
        unsafe { ffi::litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(self.ptr, index) }
    }

    /// Decode tokens per second at a given turn.
    pub fn decode_tokens_per_sec_at(&self, index: i32) -> f64 {
        unsafe { ffi::litert_lm_benchmark_info_get_decode_tokens_per_sec_at(self.ptr, index) }
    }
}

impl Drop for BenchmarkInfo {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_benchmark_info_delete(self.ptr) };
        }
    }
}

unsafe impl Send for BenchmarkInfo {}
