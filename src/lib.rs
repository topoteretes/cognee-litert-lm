//! # cognee-litert-lm
//!
//! Safe Rust bindings for the [LiteRT-LM](https://github.com/topoteretes/LiteRT-LM)
//! C++ inference engine.
//!
//! This crate provides a high-level Rust API over the LiteRT-LM C interface,
//! wrapping opaque C pointers with RAII types and safe methods.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use cognee_litert_lm::{EngineSettings, Backend, InputKind};
//!
//! // 1. Configure and create an engine
//! let settings = EngineSettings::new("path/to/model.tflite", Backend::Cpu, None, None)
//!     .expect("failed to create engine settings");
//! let engine = settings.build().expect("failed to create engine");
//!
//! // 2. Create a session and generate content
//! let session = engine.create_session(None).expect("failed to create session");
//! let responses = session
//!     .generate_content(&[InputKind::Text("Hello, world!")])
//!     .expect("generation failed");
//!
//! println!("{}", responses.first_text().unwrap());
//! ```

mod conversation;
mod engine;
mod error;
pub(crate) mod ffi;
mod session;

/// Type alias for boxed stream callbacks used internally.
pub(crate) type StreamCallbackBox = Box<dyn FnMut(&str, bool, Option<&str>) + Send>;

// Re-export the public API at the crate root.
pub use conversation::{Conversation, ConversationConfig, JsonResponse};
pub use engine::{Backend, Engine, EngineSettings};
pub use error::{Error, Result};
pub use session::{
    BenchmarkInfo, InputKind, Responses, SamplerParams, SamplerType, Session, SessionConfig,
};

/// Set the minimum log level for the LiteRT-LM library.
///
/// Log levels: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL.
pub fn set_min_log_level(level: i32) {
    unsafe { ffi::litert_lm_set_min_log_level(level) };
}
