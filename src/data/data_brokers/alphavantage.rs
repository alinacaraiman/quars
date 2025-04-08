use crate::config::Settings;
use crate::data::{HistoricalData, Record};
use crate::utils::parse_date;
use chrono::Local;
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use std::fs;

/// Alpha Vantage fetcher
pub async fn fetch_data(settings: &Settings) -> Result<HistoricalData, Box<dyn Error>> {
    // Read parameters from config under [data_api]
    let api_key = &settings.data_api.api_key;
    let tickers = &settings.data_api.tickers;
    let timeframe = settings.data_api.timeframe.to_lowercase();
    let start_date_str = &settings.data_api.start_date;
    let end_date_str = &settings.data_api.end_date;

    let start_date = parse_date(start_date_str)?;
    let end_date = parse_date(end_date_str)?;

    let function = match timeframe.as_str() {
        "daily" => "TIME_SERIES_DAILY",
        "weekly" => "TIME_SERIES_WEEKLY",
        "monthly" => "TIME_SERIES_MONTHLY",
        _ => return Err(format!("Unsupported timeframe: {}", timeframe).into()),
    };

    let client = Client::new();

    // Store all records from all tickers in a vec
    let mut all_records = Vec::new();

    for ticker in tickers {
        let url = format!(
            "https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&apikey={apikey}",
            function = function,
            symbol = ticker,
            apikey = api_key
        );

        let resp = client.get(&url).send().await?;
        let json_val: Value = resp.json().await?;

        // Save raw API result in data/raw/{ticker}/{timeframe}/{datetimenow}
        save_api_result(&json_val, ticker, &timeframe)?;
        let time_series_key = match timeframe.as_str() {
            "daily" => "Time Series (Daily)",
            "weekly" => "Weekly Time Series",
            "monthly" => "Monthly Time Series",
            _ => unreachable!(),
        };

        // Extract time series from response
        let series_obj = json_val[time_series_key]
            .as_object()
            .ok_or("Could not parse time series JSON from Alpha Vantage")?;

        for (date_str, values) in series_obj {
            if let Ok(current_date) = parse_date(date_str) {
                // Filter by date range
                if current_date < start_date || current_date > end_date {
                    continue;
                }

                let close_val = values["4. close"]
                    .as_str()
                    .ok_or("Missing close value in JSON")?;
                let close_price = close_val.parse::<f64>()?;

                all_records.push(Record {
                    date: date_str.clone().to_string(),
                    asset: ticker.to_string(),
                    price: close_price,
                });
            }
        }
    }

    Ok(all_records)
}

/// Saves the raw result in
/// data/raw/{ticker}/{timeframe}/{datetimenow}/raw.json
fn save_api_result(json_val: &Value, ticker: &str, timeframe: &str) -> Result<(), Box<dyn Error>> {
    let today = Local::now().format("%Y-%m-%d").to_string();
    let dir_path = format!("data/raw/{}/{}/{}", ticker, timeframe, today);
    fs::create_dir_all(&dir_path)?;
    let file_path = format!("{}/raw_alphavantage.json", dir_path);
    fs::write(&file_path, json_val.to_string())?;
    Ok(())
}
