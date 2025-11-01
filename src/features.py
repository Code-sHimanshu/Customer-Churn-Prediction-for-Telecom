from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

def get_column_types(df: pd.DataFrame):
    # Heuristic: numeric vs categorical
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # remove target if present
    num_cols = [c for c in num_cols if c != 'Churn']
    cat_cols = [c for c in df.columns if c not in num_cols and c != 'Churn']
    return cat_cols, num_cols

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def build_preprocessor(cat_cols, num_cols):
    # Numeric pipeline
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Returns transformed feature names in the same order as transform output.
    Works with sklearn >=1.0 get_feature_names_out.
    """
    # ColumnTransformer must be fitted
    try:
        names = preprocessor.get_feature_names_out()
        return names.tolist()
    except Exception:
        # fallback: manual extraction
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            if hasattr(transformer, 'named_steps') and 'ohe' in transformer.named_steps:
                ohe = transformer.named_steps['ohe']
                ohe_names = ohe.get_feature_names_out(cols)
                feature_names.extend(ohe_names.tolist())
            elif hasattr(transformer, 'named_steps') and 'scaler' in transformer.named_steps:
                feature_names.extend(cols)
        return feature_names

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example: average charge per month - careful with tenure=0
    if {'TotalCharges', 'tenure'}.issubset(df.columns):
        df['avg_charge_per_month'] = df['TotalCharges'] / df['tenure'].replace({0:1})
    return df

if __name__ == "__main__":
    import pandas as pd
    from data import load_raw, basic_clean
    df = load_raw()
    df = basic_clean(df)
    df = add_engineered_features(df)
    cat_cols, num_cols = get_column_types(df)
    print("categorical:", cat_cols)
    print("numeric:", num_cols)
    pre = build_preprocessor(cat_cols, num_cols)
    pre.fit(df.drop(columns=['Churn']))
    print("Feature count after transform:", pre.transform(df.drop(columns=['Churn'])).shape[1])
    print("Feature names (sample):", get_feature_names(pre)[:20])


os.makedirs('../data/processed', exist_ok=True)
processed_data.to_csv('../data/processed/processed_data.csv', index=False)
