import os
import urllib.request
import pandas as pd

def download_dataset():
    url_1 = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    url_2 = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = "data/raw/telco_churn.csv"
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        df = pd.read_csv(output_path)
    else:
        print(f"Downloading from {url_1}...")
        try:
            urllib.request.urlretrieve(url_1, output_path)
        except Exception as e:
            print(f"Failed: {e}. Trying {url_2}...")
            urllib.request.urlretrieve(url_2, output_path)
        print("Download complete.")
        df = pd.read_csv(output_path)
        
    assert "Churn" in df.columns, "Expected 'Churn' column not found!"
    print(f"Shape: {df.shape}")
    print(df.head())
    return df

if __name__ == "__main__":
    download_dataset()
