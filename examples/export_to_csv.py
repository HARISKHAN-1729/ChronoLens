#!/usr/bin/env python3
"""
Example: Convert JSON results to CSV format
"""
import json
import csv
from pathlib import Path

def json_to_csv(json_path, output_csv):
    """Convert analysis JSON to CSV format"""
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Prepare rows
    rows = []
    
    for video_data in data.get("videos", []):
        video_name = video_data.get("video_name", "unknown")
        features = video_data.get("features", {})
        
        row = {"video_name": video_name}
        
        # Flatten all numeric features
        for category, category_data in features.items():
            if isinstance(category_data, dict):
                for key, value in category_data.items():
                    if isinstance(value, (int, float)):
                        row[f"{category}_{key}"] = value
        
        rows.append(row)
    
    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"CSV exported to: {output_csv}")
        print(f"Columns: {len(fieldnames)}")
        print(f"Rows: {len(rows)}")

if __name__ == "__main__":
    json_file = "output/analysis_results.json"
    csv_file = "output/analysis_results.csv"
    
    json_to_csv(json_file, csv_file)
