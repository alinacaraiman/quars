#[cfg(feature = "data")]
pub mod config;
#[cfg(feature = "data")]
pub mod data;
pub mod bonds;
mod error;
pub mod math;
pub mod optimization;
pub mod options;
pub mod portfolio;
#[cfg(feature = "data")]
pub mod utils;
#[cfg(feature = "visualization")]
pub mod visualization;

pub use error::Error;
