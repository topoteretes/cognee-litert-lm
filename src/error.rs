use std::fmt;

/// Errors that can occur when using the LiteRT-LM library.
#[derive(Debug)]
pub enum Error {
    /// Engine creation failed (e.g. invalid model path or backend).
    EngineCreationFailed,
    /// Session creation failed.
    SessionCreationFailed,
    /// Content generation returned a null response.
    GenerationFailed,
    /// Streaming content generation failed to start.
    StreamingFailed(i32),
    /// Conversation creation failed.
    ConversationCreationFailed,
    /// Sending a message in a conversation returned null.
    SendMessageFailed,
    /// A provided string contained an interior null byte.
    NulError(std::ffi::NulError),
    /// The response index was out of bounds.
    IndexOutOfBounds { index: i32, count: i32 },
    /// Session config creation failed.
    SessionConfigCreationFailed,
    /// Conversation config creation failed.
    ConversationConfigCreationFailed,
    /// Benchmark info retrieval failed.
    BenchmarkInfoFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EngineCreationFailed => write!(f, "failed to create LiteRT-LM engine"),
            Error::SessionCreationFailed => write!(f, "failed to create LiteRT-LM session"),
            Error::GenerationFailed => write!(f, "content generation failed"),
            Error::StreamingFailed(code) => {
                write!(f, "streaming generation failed with code {code}")
            }
            Error::ConversationCreationFailed => {
                write!(f, "failed to create LiteRT-LM conversation")
            }
            Error::SendMessageFailed => write!(f, "sending message failed"),
            Error::NulError(e) => write!(f, "interior null byte in string: {e}"),
            Error::IndexOutOfBounds { index, count } => {
                write!(f, "index {index} out of bounds (count: {count})")
            }
            Error::SessionConfigCreationFailed => {
                write!(f, "failed to create session config")
            }
            Error::ConversationConfigCreationFailed => {
                write!(f, "failed to create conversation config")
            }
            Error::BenchmarkInfoFailed => write!(f, "failed to retrieve benchmark info"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NulError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Error::NulError(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
