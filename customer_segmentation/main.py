# ---------------------------------------------
# Customer Segmentation + Classification
# Project by: Nitin G Ghumare
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, classification_report
)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

def main(data_path, k, autolabel=False, threshold=50):
    # load dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded:", df.shape)

    # detect columns
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    inc_col = [c for c in df.columns if "income" in c.lower()][0]
    score_col = [c for c in df.columns if "spending" in c.lower()][0]

    # select features
    Xf = df[[age_col, inc_col, score_col]].copy()
    Xf.columns = ["Age", "Income", "Score"]

    # feature engineering
    Xf["Income_per_Age"] = Xf["Income"] / Xf["Age"]
    Xf["Age_Score"] = Xf["Age"] * Xf["Score"]

    # clustering on 3 main features
    X_cluster = StandardScaler().fit_transform(Xf[["Age","Income","Score"]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)

    # PCA plot
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_cluster)
    plt.figure()
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels, palette="Set2")
    plt.title(f"PCA (k={k})")
    plt.savefig(results_dir / f"pca_{k}.png")
    plt.close()
    print("KMeans done with k =", k)

    # find label column or autolabel
    label_col = None
    for c in df.columns:
        if c.lower() in ["label","target","class","outcome"]:
            label_col = c
            break

    if not label_col and autolabel:
        df["label"] = (df[score_col] > threshold).astype(int)
        label_col = "label"
        print(f"Auto-label created: Score > {threshold} = 1")

    # classification
    if label_col:
        y = df[label_col]

        # avoid leakage if autolabel
        if autolabel:
            X_sup = Xf.drop(columns=["Score"])
        else:
            X_sup = Xf

        X_train, X_test, y_train, y_test = train_test_split(
            X_sup, y, test_size=0.2, random_state=42, stratify=y
        )

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Logistic Regression
        lr = LogisticRegression(max_iter=400)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:,1]
        acc_lr = accuracy_score(y_test, y_pred_lr)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        auc_lr = roc_auc_score(y_test, y_prob_lr)
        report_lr = classification_report(y_test, y_pred_lr)

        # plot LR confusion
        plt.figure()
        sns.heatmap(cm_lr, annot=True, fmt="d")
        plt.title("Confusion - LR")
        plt.savefig(results_dir/"confusion_lr.png")
        plt.close()

        # SVM
        svm = SVC(probability=True)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        y_prob_svm = svm.predict_proba(X_test)[:,1]
        acc_svm = accuracy_score(y_test, y_pred_svm)
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        auc_svm = roc_auc_score(y_test, y_prob_svm)
        report_svm = classification_report(y_test, y_pred_svm)

        # plot SVM confusion
        plt.figure()
        sns.heatmap(cm_svm, annot=True, fmt="d")
        plt.title("Confusion - SVM")
        plt.savefig(results_dir/"confusion_svm.png")
        plt.close()

        # ROC curve
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
        plt.figure()
        plt.plot(fpr_lr, tpr_lr, label=f"LR AUC={auc_lr:.2f}")
        plt.plot(fpr_svm, tpr_svm, label=f"SVM AUC={auc_svm:.2f}")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.legend()
        plt.title("ROC Curve")
        plt.savefig(results_dir/"roc.png")
        plt.close()

        # print results
        print("\n--- Logistic Regression ---")
        print("Accuracy:", acc_lr)
        print("ROC AUC:", auc_lr)
        print("Confusion:\n", cm_lr)
        print(report_lr)

        print("\n--- SVM ---")
        print("Accuracy:", acc_svm)
        print("ROC AUC:", auc_svm)
        print("Confusion:\n", cm_svm)
        print(report_svm)

        # save to file
        with open(results_dir/"classification_metrics.txt","w") as f:
            f.write("Logistic Regression\n")
            f.write(f"Accuracy={acc_lr}, AUC={auc_lr}\n")
            f.write(str(cm_lr)+"\n")
            f.write(report_lr+"\n\n")
            f.write("SVM\n")
            f.write(f"Accuracy={acc_svm}, AUC={auc_svm}\n")
            f.write(str(cm_svm)+"\n")
            f.write(report_svm)

    else:
        print("No label column found, skipping classification.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data","-d",type=str,required=True)
    parser.add_argument("--k","-k",type=int,default=5)
    parser.add_argument("--autolabel",action="store_true")
    parser.add_argument("--threshold",type=int,default=50)
    args = parser.parse_args()
    main(args.data,args.k,autolabel=args.autolabel,threshold=args.threshold)
