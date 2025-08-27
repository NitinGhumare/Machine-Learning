############################################################
# Author      : Nitin G Ghumare
# Date        : 27-08-2025
# Project     : Play Predictor (Case Study 1)
# Description : Machine Learning project using KNN to predict
#               whether to play or not based on weather dataset.
#               Script follows industry-standard function flow.
############################################################

# ========== Import Libraries ==========
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ========== Step 1: Load Dataset ==========
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"✅ Dataset loaded. Shape = {df.shape}")
    return df


# ========== Step 2: Preprocess Data ==========
def preprocess_data(df: pd.DataFrame):
    """
    Encode categorical columns and split into features & target.
    """
    encoder = LabelEncoder()
    for col in df.columns:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop(columns=["Play"])
    y = df["Play"]

    print("✅ Data preprocessing completed.")
    return X, y


# ========== Step 3: Train Model ==========
def train_model(X, y, k: int = 3):
    """
    Train KNN model with given K.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    return model, X_test, y_test


# ========== Step 4: Evaluate Model ==========
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using accuracy, precision, recall, confusion matrix.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Evaluation Metrics")
    print(f"   Accuracy  : {acc:.2f}")
    print(f"   Precision : {prec:.2f}")
    print(f"   Recall    : {rec:.2f}")
    print(f"   Confusion Matrix:\n{cm}")
    print("   Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return acc


# ========== Step 5: Save Model ==========
def save_model(model, save_dir: str = "models/", filename: str = "knn_play.joblib"):
    """
    Save trained model using joblib.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(save_dir) / filename
    joblib.dump(model, filepath)

    print(f"💾 Model saved at: {filepath}")


# ========== Step 6: Main Function ==========
def main():
    """
    Main workflow function.
    """
    # Load
    df = load_dataset("PlayPredictor.csv")

    # Preprocess
    X, y = preprocess_data(df)

    # Loop for K=1 to 10
    best_acc = 0
    best_model = None
    best_k = None

    for k in range(1, 11):
        print("\n===================================================")
        print(f"🔹 Training model with K = {k}")

        model, X_test, y_test = train_model(X, y, k=k)
        acc = evaluate_model(model, X_test, y_test)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k = k

    # Save best model
    if best_model:
        save_model(best_model, filename=f"knn_play_bestK{best_k}.joblib")
        print(f"\n🏆 Best K = {best_k} with Accuracy = {best_acc:.2f}")


# ========== Script Entry ==========
if __name__ == "__main__":
    main()
