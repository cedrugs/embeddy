use std::fmt;

#[derive(Debug)]
pub enum Error {
    ModelNotFound(String),
    ModelLoadFailed(String),
    InvalidInput(String),
    DownloadFailed(String),
    Config(String),
    Embedding(String),
    Io(std::io::Error),
    Serialization(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            Error::ModelLoadFailed(msg) => write!(f, "Failed to load model: {}", msg),
            Error::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Error::DownloadFailed(msg) => write!(f, "Download failed: {}", msg),
            Error::Config(msg) => write!(f, "Configuration error: {}", msg),
            Error::Embedding(msg) => write!(f, "Embedding error: {}", msg),
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Serialization(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<toml::de::Error> for Error {
    fn from(err: toml::de::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<toml::ser::Error> for Error {
    fn from(err: toml::ser::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
