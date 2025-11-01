import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ----------------------------
# 1️⃣ Define Paths
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'Telco-Customer-Churn.csv')
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'processed_data.csv')

os.makedirs(os.path.join(ROOT_DIR, 'data', 'processed'), exist_ok=True)

# ----------------------------
# 2️⃣ Preprocess Data
# ----------------------------
def preprocess_data():
    print("🔄 Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Raw data shape: {df.shape}")

    # Drop customerID (not useful for modeling)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert target column 'Churn' to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Handle TotalCharges (sometimes blank)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    print("⚙️ Transforming data...")
    X_processed = preprocessor.fit_transform(X)

    # Get new column names from OneHotEncoder
    cat_features = list(preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(cat_cols))
    processed_cols = num_cols + cat_features

    # Combine into one DataFrame
    processed_df = pd.DataFrame(X_processed, columns=processed_cols)
    processed_df['Churn'] = y.values

    # Save processed data
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"📊 Final shape: {processed_df.shape}")

# ----------------------------
# 3️⃣ Run the Preprocessing
# ----------------------------
if __name__ == "__main__":
    preprocess_data()
