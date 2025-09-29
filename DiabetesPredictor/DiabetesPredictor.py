############################################################
# Author      : Nitin G Ghumare
# Email       : nghumare570@gmail.com
# Date        : 28-09-2025
# Project     : Diabetes Detection
# Description : Machine Learning pipeline with Logistic Regression & SVM
#               Includes EDA, preprocessing, training, evaluation,
#               and saves results in results/ folder.
############################################################

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# -----------------------
# Config
# -----------------------
DATA_PATH = "DiabetesDataset.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "Outcome"
INVALID_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def DiabetesPredictor(Datapath):
    df = pd.read_csv(Datapath)

    print("✅ Dataset loaded succesfully : ")
    print(df.head())
    print("📏 Dimensions of dataset : ", df.shape)

    # Replace invalid zeros with median
    for col in INVALID_ZERO_COLS:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
            df[col] = df[col].fillna(df[col].median())

    print("📏 Dimensions after cleaning : ", df.shape)

    # ---------- EDA plots ----------
    plt.figure()
    sns.countplot(data=df, x=TARGET_COL).set_title("Outcome Distribution")
    plt.savefig(RESULTS_DIR / "plot_outcome_count.png")
    plt.close()

    if "Age" in df.columns:
        plt.figure()
        df["Age"].plot.hist().set_title("Age Distribution")
        plt.savefig(RESULTS_DIR / "plot_age_hist.png")
        plt.close()

    if "Glucose" in df.columns:
        plt.figure()
        df["Glucose"].plot.hist().set_title("Glucose Distribution")
        plt.savefig(RESULTS_DIR / "plot_glucose_hist.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(RESULTS_DIR / "plot_corr_heatmap.png")
    plt.close()

    # ---------- Prepare features ----------
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scale, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---------- Logistic Regression ----------
    lr_model = LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
    lr_model.fit(x_train, y_train)
    y_pred_lr = lr_model.predict(x_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    y_prob_lr = lr_model.predict_proba(x_test)[:, 1]
    roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

    # ---------- SVM ----------
    svm_model = SVC(probability=True, random_state=RANDOM_STATE)
    svm_model.fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    y_prob_svm = svm_model.predict_proba(x_test)[:, 1]
    roc_auc_svm = roc_auc_score(y_test, y_prob_svm)

    # ---------- Save Confusion Matrix plots ----------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix (Logistic Regression)")
    plt.savefig(RESULTS_DIR / "plot_confusion_matrix_lr.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title("Confusion Matrix (SVM)")
    plt.savefig(RESULTS_DIR / "plot_confusion_matrix_svm.png")
    plt.close()

    # ---------- Save ROC curves ----------
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_lr, tpr_lr, label=f"LR AUC = {roc_auc_lr:.4f}")
    plt.plot(fpr_svm, tpr_svm, label=f"SVM AUC = {roc_auc_svm:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(RESULTS_DIR / "plot_roc_curve.png")
    plt.close()

    # ---------- Save metrics ----------
    with open(RESULTS_DIR / "metrics.txt", "w") as fh:
        fh.write(f"Logistic Regression Accuracy: {acc_lr:.4f}\n")
        fh.write(f"Logistic Regression ROC AUC: {roc_auc_lr:.4f}\n")
        fh.write(f"Confusion Matrix (LR):\n{cm_lr}\n\n")
        fh.write(f"SVM Accuracy: {acc_svm:.4f}\n")
        fh.write(f"SVM ROC AUC: {roc_auc_svm:.4f}\n")
        fh.write(f"Confusion Matrix (SVM):\n{cm_svm}\n")

    print("🎯 Logistic Regression Accuracy : ", acc_lr)
    print("📊 Confusion matrix (LR) :\n", cm_lr)
    print("📈 ROC AUC (LR) : ", roc_auc_lr)

    print("\n🎯 SVM Accuracy : ", acc_svm)
    print("📊 Confusion matrix (SVM) :\n", cm_svm)
    print("📈 ROC AUC (SVM) : ", roc_auc_svm)

    print(f"\n✅ All results saved in: {RESULTS_DIR.resolve()}")


def main():
    DiabetesPredictor(DATA_PATH)


if __name__ == "__main__":
    main()
