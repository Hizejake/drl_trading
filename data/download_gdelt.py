import os
import requests
import zipfile
import pandas as pd
from datetime import datetime
import io

# We will download the latest available GDELT GKG file
# GDELT GKG updates every 15 minutes.
GDELT_LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")

def download_latest_gdelt():
    print("Fetching URL for latest GDELT GKG dataset...")
    response = requests.get(GDELT_LAST_UPDATE_URL)
    response.raise_for_status()
    
    # The lastupdate.txt file has format: size  hash  url
    # We want the GKG dataset (usually the third line)
    lines = response.text.strip().split('\n')
    gkg_url = None
    for line in lines:
        if "gkg.csv.zip" in line:
            gkg_url = line.split()[-1]
            break
            
    if not gkg_url:
        print("Could not find GKG URL in lastupdate.txt")
        return None

    print(f"Downloading {gkg_url}...")
    r = requests.get(gkg_url)
    r.raise_for_status()
    
    print("Extracting CSV...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # there should be only one file in the zip
        csv_filename = z.namelist()[0]
        z.extract(csv_filename, DATA_DIR)
        
    extracted_path = os.path.join(DATA_DIR, csv_filename)
    print(f"Downloaded and extracted to: {extracted_path}")
    
    # Filter for Apple (AAPL) or Amazon (AMZN) mentions as a sample
    print("Filtering for US Equity mentions (AAPL, AMZN, MSFT, TSLA, NVDA)...")
    # GKG files have no header by default in V2, but we mostly care about V2GKGRecord
    # Columns: GKGRECORDID, V2.1DATE, V2SOURCECOLLECTIONIDENTIFIER, V2SOURCECOMMONNAME, V2DOCUMENTIDENTIFIER, ...
    # We'll do a simple string match for now to keep it lightweight
    
    filtered_rows = []
    target_companies = ["Apple", "Amazon", "Microsoft", "Tesla", "Nvidia"]
    
    with open(extracted_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if any(comp in line for comp in target_companies):
                filtered_rows.append(line)
                
    filtered_path = os.path.join(DATA_DIR, "filtered_gkg_sample.csv")
    with open(filtered_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_rows)
        
    print(f"Filtered {len(filtered_rows)} potentially relevant events saved to {filtered_path}")
    return filtered_path

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_latest_gdelt()
