import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ------------------------------
# 1️⃣ Define Paths
# ------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "processed_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

# ------------------------------
# 2️⃣ Load Data
# ------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y

# ------------------------------
# 3️⃣ Evaluate the Model
# ------------------------------
def evaluate_model(model_path):
    print(f"📦 Loading model from: {model_path}")
    model = joblib.load(model_path)

    X, y = load_data()
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    print("\n📊 Model Performance on Full Data:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    # ------------------------------
    # 4️⃣ Confusion Matrix Plot
    # ------------------------------
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    plot_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"🖼️ Confusion matrix saved to: {plot_path}")
    plt.close()

    # ------------------------------
    # 5️⃣ Feature Importance (XGBoost Only)
    # ------------------------------
    try:
        importances = model.named_steps['model'].feature_importances_
        feature_names = model.named_steps['scaler'].get_feature_names_out()
    except Exception:
        try:
            importances = model.named_steps['model'].feature_importances_
            feature_names = X.columns
        except Exception:
            print("⚠️ Could not extract feature importances.")
            return

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(15))
    plt.title("Top 15 Feature Importances (XGBoost)")
    plt.tight_layout()

    fi_path = os.path.join(REPORT_DIR, "feature_importance.png")
    plt.savefig(fi_path)
    print(f"📊 Feature importance chart saved to: {fi_path}")
    plt.close()

# ------------------------------
# 6️⃣ Run Evaluation
# ------------------------------
if __name__ == "__main__":
    latest_model = sorted(
        [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
        key=os.path.getmtime,
        reverse=True
    )[0]

    evaluate_model(latest_model)
