use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
};

use chrono::{NaiveDate, ParseError};
use csv::WriterBuilder;

use crate::data::HistoricalData;

/// Writes a HistoricalData to CSV
pub fn write_to_csv(data: &HistoricalData, output_path: &str) -> Result<(), csv::Error> {
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories for CSV output");
    }
    let mut date_set = BTreeSet::new();
    let mut asset_set = BTreeSet::new();
    for record in data {
        date_set.insert(record.date.clone());
        asset_set.insert(record.asset.clone());
    }
    let dates: Vec<String> = date_set.into_iter().collect();
    let assets: Vec<String> = asset_set.into_iter().collect();
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(output_path)?;

    let mut header = vec!["date".to_string()];
    header.extend(assets.iter().cloned());
    wtr.write_record(&header)?;

    let mut lookup: HashMap<(String, String), f64> = HashMap::new();
    for record in data {
        lookup.insert((record.date.clone(), record.asset.clone()), record.price);
    }

    for date in dates {
        let mut row = vec![date.clone()];
        for asset in &assets {
            // If no price found, leave the cell blank.
            if let Some(price) = lookup.get(&(date.clone(), asset.clone())) {
                row.push(price.to_string());
            } else {
                row.push("".to_string());
            }
        }
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn parse_date(date_str: &str) -> Result<NaiveDate, ParseError> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
}
