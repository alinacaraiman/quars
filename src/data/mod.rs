pub mod data_brokers;

use crate::config::Settings;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
pub struct Record {
    pub date: String,
    pub asset: String,
    pub price: f64,
}

pub type HistoricalData = Vec<Record>;

/// Reads CSV into HistoricalData
fn read_csv(path: &str) -> Result<HistoricalData, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let date = record.get(0).unwrap().to_string();
        for (i, asset_name) in headers.iter().enumerate().skip(1) {
            if let Some(price_str) = record.get(i) {
                if let Ok(price) = price_str.parse::<f64>() {
                    data.push(Record {
                        date: date.clone(),
                        asset: asset_name.to_string(),
                        price,
                    });
                }
            }
        }
    }
    Ok(data)
}

/// Main Alpha Vantage fetcher
pub async fn fetch_data(settings: &Settings) -> Result<HistoricalData, Box<dyn Error>> {
    match settings.general.data_source.as_str() {
        "csv" => read_csv(&settings.general.data_file),
        "api" => data_brokers::fetch_data(settings).await,
        _ => Err("Unknown data source specified.".into()),
    }
}
