use crate::config::Settings;
use crate::data::{HistoricalData, Record};
use crate::utils;
use chrono::DateTime;
use reqwest::Client;
use serde_json::Value;
use crate::Error;

/// Fetch adjusted closes from the Yahoo Finance chart API (unofficial, no API key)
pub async fn fetch_data(settings: &Settings) -> Result<HistoricalData, Error> {
    let api = &settings.data_api;
    let interval = match api.timeframe.to_lowercase().as_str() {
        "daily" => "1d",
        "weekly" => "1wk",
        "monthly" => "1mo",
        timeframe => return Err(Error::Data(format!("Unsupported timeframe: {}", timeframe))),
    };
    let timestamp = |date: &str| {
        utils::parse_date(date).map(|d| d.and_time(Default::default()).and_utc().timestamp())
    };
    let (start, end) = (
        timestamp(&api.start_date)?,
        timestamp(&api.end_date)? + 86_400,
    );

    // Yahoo rejects requests without a user agent
    let client = Client::builder().user_agent("Mozilla/5.0").build()?;
    let mut records = Vec::new();
    for ticker in &api.tickers {
        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}",
            ticker, start, end, interval
        );
        let json: Value = client.get(&url).send().await?.json().await?;
        let result = &json["chart"]["result"][0];
        let (Some(timestamps), Some(closes)) = (
            result["timestamp"].as_array(),
            result["indicators"]["adjclose"][0]["adjclose"].as_array(),
        ) else {
            return Err(Error::Data(format!(
                "Error from Yahoo Finance for {}: {}",
                ticker, json["chart"]["error"]
            )));
        };
        for (t, close) in timestamps.iter().zip(closes) {
            if let (Some(t), Some(price)) = (t.as_i64(), close.as_f64()) {
                let date =
                    DateTime::from_timestamp(t, 0).ok_or(Error::Data("Invalid timestamp from Yahoo Finance".into()))?;
                records.push(Record {
                    date: date.format("%Y-%m-%d").to_string(),
                    asset: ticker.clone(),
                    price,
                });
            }
        }
    }
    Ok(records)
}
