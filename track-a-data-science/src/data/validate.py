import pandas as pd

def validate_dataset(df: pd.DataFrame) -> dict:
    report = {
        "is_empty": df.empty,
        "has_churn_col": "Churn" in df.columns,
        "missing_values": df.isnull().sum().to_dict(),
        "shape": df.shape
    }
    
    if df.empty:
        raise ValueError("Dataset is empty!")
    if "Churn" not in df.columns:
        raise ValueError("Missing 'Churn' column!")
        
    return report

if __name__ == "__main__":
    df = pd.read_csv("data/raw/telco_churn.csv")
    print(validate_dataset(df))
