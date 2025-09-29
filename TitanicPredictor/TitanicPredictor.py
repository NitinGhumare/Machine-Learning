############################################################
# Author      : Nitin G Ghumare
# Email       : nghumare570@gmail.com
# Date        : 28-09-2025
# Project     : Titanic Survival Predictor
# Description : Minimal changes to original MarvellousTitanicLogistic.
#               Saves EDA plots, confusion matrix and ROC curve inside
#               a results/ folder and writes metrics.txt.
############################################################

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import countplot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# -----------------------
# Config
# -----------------------
DATA_PATH = "MarvellousTitanicDataset.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2

def MarvellousTitanicLogistic(Datapath):
    df = pd.read_csv(Datapath)

    print("✅ Dataset loaded succesfully : ")
    print(df.head())

    print("📏 Dimensions of dataset : ", df.shape)

    # Drop unwanted columns if present (keeps your original logic)
    for col in ['Passengerid', 'zero']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print("📏 Dimensions after drop (if any) : ", df.shape)

    # Safe imputation for Embarked (avoid chained assignment warnings)
    if 'Embarked' in df.columns:
        df.loc[:, 'Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # ---------- EDA plots (saved into results/) ----------
    target = "Survived"

    plt.figure()
    countplot(data=df, x=target).set_title("Survived vs Non Survived")
    plt.savefig(RESULTS_DIR / "plot_survived_count.png")
    plt.close()

    if 'Sex' in df.columns:
        plt.figure()
        countplot(data=df, x=target, hue='Sex').set_title("Based on gender")
        plt.savefig(RESULTS_DIR / "plot_survived_gender.png")
        plt.close()

    if 'Pclass' in df.columns:
        plt.figure()
        countplot(data=df, x=target, hue='Pclass').set_title("Based on Pclass")
        plt.savefig(RESULTS_DIR / "plot_survived_pclass.png")
        plt.close()

    if 'Age' in df.columns:
        plt.figure()
        df['Age'].plot.hist().set_title("Age report")
        plt.savefig(RESULTS_DIR / "plot_age_hist.png")
        plt.close()

    if 'Fare' in df.columns:
        plt.figure()
        df['Fare'].plot.hist().set_title("Fare report")
        plt.savefig(RESULTS_DIR / "plot_fare_hist.png")
        plt.close()

    # Correlation heatmap (only numeric columns)
    plt.figure(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.savefig(RESULTS_DIR / "plot_corr_heatmap.png")
    plt.close()

    # ---------- Prepare features ----------
    # Keep same API as your original code but ensure categorical -> numeric
    x = df.drop(columns=[target])
    y = df[target]

    # Convert categorical features to numeric with one-hot (keeps original behavior mostly,
    # but prevents scaler errors when non-numeric columns exist)
    x = pd.get_dummies(x, drop_first=True)

    print("📏 Dimensions of Features : ", x.shape)
    print("📏 Dimensions of Labels : ", y.shape)

    # ---------- Scaling + Train/Test ----------
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scale, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ---------- Model ----------
    model = LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Try to get probabilities for ROC AUC (works for LogisticRegression)
    y_prob = None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
    except Exception:
        y_prob = None

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = None

    # ---------- Save Confusion Matrix plot ----------
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(RESULTS_DIR / "plot_confusion_matrix.png")
    plt.close()

    # Save confusion matrix numbers as txt
    with open(RESULTS_DIR / "confusion_matrix.txt", "w") as fh:
        fh.write("Confusion matrix (rows=actual, cols=predicted):\n")
        fh.write(np.array2string(cm))
        fh.write("\n")

    # ---------- Save ROC curve plot ----------
    if y_prob is not None and roc_auc is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0,1], [0,1], linestyle='--', color='grey')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc='lower right')
        plt.savefig(RESULTS_DIR / "plot_roc_curve.png")
        plt.close()

    # ---------- Save metrics ----------
    metrics_txt = RESULTS_DIR / "metrics.txt"
    with open(metrics_txt, "w") as fh:
        fh.write(f"Accuracy: {accuracy:.4f}\n")
        if roc_auc is not None:
            fh.write(f"ROC AUC: {roc_auc:.4f}\n")
        fh.write("Confusion matrix:\n")
        fh.write(np.array2string(cm))
        fh.write("\n")

    print("🎯 Accuracy is : ", accuracy)
    print("📊 Confusion matrix : ")
    print(cm)
    if roc_auc is not None:
        print("📈 ROC AUC : ", roc_auc)

    print(f"\n✅ All important plots and metrics saved inside: {RESULTS_DIR.resolve()}")

def main():
    MarvellousTitanicLogistic(DATA_PATH)

if __name__ == "__main__":
    main()
