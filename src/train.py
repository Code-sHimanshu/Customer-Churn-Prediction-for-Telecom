import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ------------------------------
# 1️⃣ Define Paths
# ------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "processed_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------------
# 2️⃣ Load Processed Data
# ------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = shuffle(df, random_state=42)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# 3️⃣ Define Model Pipeline
# ------------------------------
def get_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    return pipeline

# ------------------------------
# 4️⃣ Hyperparameter Grid for Tuning
# ------------------------------
def get_param_dist():
    param_dist = {
        'model__n_estimators': [100, 200, 300, 400, 500],
        'model__max_depth': [3, 5, 7, 10],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],
        'model__gamma': [0, 0.1, 0.2],
        'model__scale_pos_weight': [1, 2, 3]  # For imbalance
    }
    return param_dist

# ------------------------------
# 5️⃣ Train Model with Cross Validation + Tuning
# ------------------------------
def train_model():
    print("🚀 Starting XGBoost training with hyperparameter tuning and CV...")

    X_train, X_test, y_train, y_test = load_data()

    # Handle class imbalance using SMOTE
    print("⚖️ Applying SMOTE to handle imbalance...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: {X_train_res.shape}, Positive class ratio: {y_train_res.mean():.2f}")

    pipeline = get_pipeline()
    param_dist = get_param_dist()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("🔍 Running randomized search for best hyperparameters...")
    random_search.fit(X_train_res, y_train_res)

    print(f"\n✅ Best Parameters: {random_search.best_params_}")
    print(f"🏆 Best Cross-Validation F1 Score: {random_search.best_score_:.4f}")

    # Evaluate on test data
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n📊 Final Evaluation on Test Data:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # ------------------------------
    # 6️⃣ Save Model and Reports
    # ------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"xgb_best_model_{timestamp}.pkl")
    report_path = os.path.join(REPORT_DIR, f"xgb_evaluation_report_{timestamp}.json")

    joblib.dump(best_model, model_path)
    with open(report_path, "w") as f:
        json.dump({
            "best_params": random_search.best_params_,
            "cv_best_score": random_search.best_score_,
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }, f, indent=4)

    print(f"\n💾 Model saved to: {model_path}")
    print(f"📝 Evaluation report saved to: {report_path}")

# ------------------------------
# 7️⃣ Run Script
# ------------------------------
if __name__ == "__main__":
    train_model()
