use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Arguments or series a function cannot work with
    InvalidInput(String),
    /// Reading, fetching or parsing data
    Data(String),
    /// Drawing a chart
    Plot(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Error::Data(message) => write!(f, "data: {message}"),
            Error::Plot(message) => write!(f, "plot: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Error::InvalidInput(message.into())
    }
}

#[cfg(feature = "openblas")]
impl From<ndarray_linalg::error::LinalgError> for Error {
    fn from(error: ndarray_linalg::error::LinalgError) -> Self {
        Error::InvalidInput(error.to_string())
    }
}

#[cfg(feature = "data")]
macro_rules! data_error {
    ($($source:ty),*) => {$(
        impl From<$source> for Error {
            fn from(error: $source) -> Self {
                Error::Data(error.to_string())
            }
        }
    )*};
}
#[cfg(feature = "data")]
data_error!(
    std::io::Error,
    std::num::ParseFloatError,
    csv::Error,
    reqwest::Error,
    chrono::ParseError
);

#[cfg(feature = "visualization")]
impl<E: std::error::Error + Send + Sync> From<plotters::drawing::DrawingAreaErrorKind<E>>
    for Error
{
    fn from(error: plotters::drawing::DrawingAreaErrorKind<E>) -> Self {
        Error::Plot(error.to_string())
    }
}
