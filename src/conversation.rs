use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::ptr;

use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::ffi;
use crate::session::{BenchmarkInfo, SessionConfig};

/// Configuration for creating a [`Conversation`].
pub struct ConversationConfig {
    ptr: *mut ffi::LiteRtLmConversationConfig,
}

impl ConversationConfig {
    /// Create a new conversation config.
    ///
    /// - `engine`: the engine to use.
    /// - `session_config`: optional session config (uses defaults if `None`).
    /// - `system_message_json`: optional system message in JSON format.
    /// - `tools_json`: optional tools description in JSON array format.
    /// - `messages_json`: optional initial messages in JSON format.
    /// - `enable_constrained_decoding`: whether to enable constrained decoding.
    pub fn new(
        engine: &Engine,
        session_config: Option<&SessionConfig>,
        system_message_json: Option<&str>,
        tools_json: Option<&str>,
        messages_json: Option<&str>,
        enable_constrained_decoding: bool,
    ) -> Result<Self> {
        let c_system = system_message_json.map(CString::new).transpose()?;
        let c_tools = tools_json.map(CString::new).transpose()?;
        let c_messages = messages_json.map(CString::new).transpose()?;

        let ptr = unsafe {
            ffi::litert_lm_conversation_config_create(
                engine.ptr,
                session_config.map_or(ptr::null(), |c| c.ptr as *const _),
                c_system.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                c_tools.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                c_messages.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                enable_constrained_decoding,
            )
        };

        if ptr.is_null() {
            return Err(Error::ConversationConfigCreationFailed);
        }
        Ok(Self { ptr })
    }
}

impl Drop for ConversationConfig {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_conversation_config_delete(self.ptr) };
        }
    }
}

unsafe impl Send for ConversationConfig {}

/// A multi-turn conversation backed by a LiteRT-LM engine.
pub struct Conversation {
    ptr: *mut ffi::LiteRtLmConversation,
}

impl Conversation {
    /// Create a new conversation from an engine with optional config.
    pub fn new(engine: &Engine, config: Option<ConversationConfig>) -> Result<Self> {
        let config_ptr = config.as_ref().map_or(ptr::null_mut(), |c| c.ptr);

        let ptr = unsafe { ffi::litert_lm_conversation_create(engine.ptr, config_ptr) };
        if ptr.is_null() {
            return Err(Error::ConversationCreationFailed);
        }
        Ok(Self { ptr })
    }

    /// Send a message (as a JSON string) and get a JSON response (blocking).
    pub fn send_message(&self, message_json: &str) -> Result<JsonResponse> {
        let c_msg = CString::new(message_json)?;
        let resp_ptr =
            unsafe { ffi::litert_lm_conversation_send_message(self.ptr, c_msg.as_ptr()) };
        if resp_ptr.is_null() {
            return Err(Error::SendMessageFailed);
        }
        Ok(JsonResponse { ptr: resp_ptr })
    }

    /// Send a message and stream the response via a callback (non-blocking).
    ///
    /// The callback receives `(chunk, is_final, error_message)`.
    pub fn send_message_stream<F>(&self, message_json: &str, callback: F) -> Result<()>
    where
        F: FnMut(&str, bool, Option<&str>) + Send + 'static,
    {
        let c_msg = CString::new(message_json)?;
        let callback_box: Box<Box<dyn FnMut(&str, bool, Option<&str>) + Send>> =
            Box::new(Box::new(callback));
        let callback_data = Box::into_raw(callback_box) as *mut c_void;

        let ret = unsafe {
            ffi::litert_lm_conversation_send_message_stream(
                self.ptr,
                c_msg.as_ptr(),
                Some(conversation_stream_trampoline),
                callback_data,
            )
        };

        if ret != 0 {
            unsafe {
                drop(Box::from_raw(
                    callback_data as *mut Box<dyn FnMut(&str, bool, Option<&str>) + Send>,
                ));
            }
            return Err(Error::StreamingFailed(ret));
        }
        Ok(())
    }

    /// Cancel an ongoing inference process.
    pub fn cancel(&self) {
        unsafe { ffi::litert_lm_conversation_cancel_process(self.ptr) };
    }

    /// Get benchmark info from the conversation.
    pub fn get_benchmark_info(&self) -> Result<BenchmarkInfo> {
        let ptr = unsafe { ffi::litert_lm_conversation_get_benchmark_info(self.ptr) };
        if ptr.is_null() {
            return Err(Error::BenchmarkInfoFailed);
        }
        Ok(BenchmarkInfo { ptr })
    }
}

impl Drop for Conversation {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_conversation_delete(self.ptr) };
        }
    }
}

unsafe impl Send for Conversation {}

/// C-compatible trampoline for conversation streaming callback.
unsafe extern "C" fn conversation_stream_trampoline(
    callback_data: *mut c_void,
    chunk: *const std::os::raw::c_char,
    is_final: bool,
    error_msg: *const std::os::raw::c_char,
) {
    let cb = &mut *(callback_data as *mut Box<dyn FnMut(&str, bool, Option<&str>) + Send>);

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

    if is_final {
        drop(Box::from_raw(
            callback_data as *mut Box<dyn FnMut(&str, bool, Option<&str>) + Send>,
        ));
    }
}

/// A JSON response from a conversation.
pub struct JsonResponse {
    ptr: *mut ffi::LiteRtLmJsonResponse,
}

impl JsonResponse {
    /// Get the JSON string from the response.
    pub fn as_str(&self) -> Option<&str> {
        let ptr = unsafe { ffi::litert_lm_json_response_get_string(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().ok()
    }
}

impl Drop for JsonResponse {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::litert_lm_json_response_delete(self.ptr) };
        }
    }
}

unsafe impl Send for JsonResponse {}
