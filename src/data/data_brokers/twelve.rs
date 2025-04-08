use crate::config::Settings;
use crate::data::{HistoricalData, Record};
use crate::utils;
use chrono::{Local, NaiveDate};
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use std::fs;

/// Fetch data from Twelve Data API
pub async fn fetch_data(settings: &Settings) -> Result<HistoricalData, Box<dyn Error>> {
    // Read parameters from the [data_api] section.
    let api_key = &settings.data_api.api_key;
    let tickers = &settings.data_api.tickers;
    let timeframe = settings.data_api.timeframe.to_lowercase(); // Expected "daily", "1min", etc.
    let start_date_str = &settings.data_api.start_date;
    let end_date_str = &settings.data_api.end_date;

    let start_date = utils::parse_date(start_date_str)?;
    let end_date = utils::parse_date(end_date_str)?;

    // TODO: Expand to intraday
    let tf_twelve = match timeframe.as_str() {
        "daily" => "1day",
        "weekly" => "1week",
        "monthly" => "1month",
        _ => return Err(format!("Unsupported timeframe: {}", timeframe).into()),
    };

    let base_url = "https://api.twelvedata.com/time_series";

    let client = Client::new();
    let mut all_records = Vec::new();

    for ticker in tickers {
        let url = format!(
            "{}?symbol={}&interval={}&outputsize=5000&apikey={}",
            base_url, ticker, tf_twelve, api_key
        );

        let resp = client.get(&url).send().await?;
        let json_val: Value = resp.json().await?;

        save_api_result(&json_val, ticker, &timeframe)?;

        if let Some(status) = json_val.get("status") {
            if status == "error" {
                return Err(
                    format!("Error from Twelve Data API for {}: {:?}", ticker, json_val).into(),
                );
            }
        }

        let values = json_val
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or("Could not parse 'values' array from Twelve response")?;

        for entry in values {
            if let Some(date_str) = entry.get("datetime").and_then(|v| v.as_str()) {
                // Twelve returns datetime such as "2020-02-26 15:59:00"
                // Extract the date part
                let date_part = &date_str[..10];
                if let Ok(current_date) = NaiveDate::parse_from_str(date_part, "%Y-%m-%d") {
                    if current_date < start_date || current_date > end_date {
                        continue;
                    }

                    if let Some(close_str) = entry.get("close").and_then(|v| v.as_str()) {
                        let close_price = close_str.parse::<f64>()?;
                        all_records.push(Record {
                            date: date_str.to_string(),
                            asset: ticker.to_string(),
                            price: close_price,
                        });
                    }
                }
            }
        }
    }

    Ok(all_records)
}

/// Save the raw JSON API result in
/// data/raw/{ticker}/{timeframe}/{datetimenow}/raw.json
fn save_api_result(json_val: &Value, ticker: &str, timeframe: &str) -> Result<(), Box<dyn Error>> {
    let today = Local::now().format("%Y-%m-%d").to_string();
    let dir_path = format!("data/raw/{}/{}/{}", ticker, timeframe, today);
    fs::create_dir_all(&dir_path)?;
    let file_path = format!("{}/raw.json", dir_path);
    fs::write(&file_path, json_val.to_string())?;
    Ok(())
}
