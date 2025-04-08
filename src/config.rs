use config::{Config, File};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub general: General,
    pub data_api: DataAPI,
    pub portofolio_optimization: PortofolioOptimization,
}

#[derive(Debug, Deserialize)]
pub struct General {
    pub data_source: String,
    pub data_file: String,
}

#[derive(Debug, Deserialize)]
pub struct PortofolioOptimization {
    pub method: String,
    pub sub_method: String,
    pub risk_free_rate: f64,
    pub params: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct DataAPI {
    pub source: String,
    pub api_key: String,
    pub tickers: Vec<String>,
    pub start_date: String,
    pub end_date: String,
    pub timeframe: String,
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        dotenv::dotenv().ok();
        let s = Config::builder()
            .add_source(File::with_name("config"))
            // Retrieve the api key from .env
            .add_source(config::Environment::with_prefix("APP").separator("__"))
            .build()?;
        s.try_deserialize()
    }
}
