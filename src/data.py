"""
Load and clean Telco dataset.
Assumes CSV at data/raw/telco_churn.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/telco_churn.csv"

def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges sometimes read as string - coerce to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop rows with missing TotalCharges (or you can impute)
    if df['TotalCharges'].isna().sum() > 0:
        df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    # Map target to binary
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    # Remove customerID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df

def train_test_split_stratified(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

if __name__ == "__main__":
    df = load_raw()
    print("Raw shape:", df.shape)
    df = basic_clean(df)
    print("Clean shape:", df.shape)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)
    print("Train/Test sizes:", X_train.shape, X_test.shape)
