pub mod alphavantage;
pub mod twelve;
use crate::config::Settings;

use super::HistoricalData;

pub async fn fetch_data(settings: &Settings) -> Result<HistoricalData, Box<dyn std::error::Error>> {
    match settings.data_api.source.to_lowercase().as_str() {
        "alphavantage" => alphavantage::fetch_data(settings).await,
        "twelve" => twelve::fetch_data(settings).await,
        _ => Err("Unsupported data broker specified. Please open an issue, specifying your data broker and useful links.".into()),
    }
}
