# 🛍️ Customer Segmentation and Classification

This project performs **customer segmentation** using KMeans clustering and evaluates simple **classification models** (Logistic Regression and SVM).

---

## 📂 Dataset
Dataset used: `mall_customers_full.csv` with 200 rows and 5 columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 🔑 Features
- **KMeans clustering** (Age, Income, Spending Score)
- **PCA visualization** for clusters
- **Feature Engineering**
  - `Income_per_Age = Income / Age`
  - `Age_Score = Age * Score`
- **Classification Models**
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Evaluation**
  - Accuracy
  - Confusion Matrix
  - ROC AUC
  - Classification Report (Precision, Recall, F1)

---

## 📊 Results
All outputs are saved inside the `results/` folder:
- `pca_k5.png` – PCA scatter plot  
- `confusion_lr.png` – Confusion Matrix (Logistic Regression)  
- `confusion_svm.png` – Confusion Matrix (SVM)  
- `roc.png` – ROC Curve comparison  
- `classification_metrics.txt` – Accuracy, AUC, and classification reports  

---

## 🛠️ How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Run clustering only
```bash
python main.py --data mall_customers_full.csv --k 5
```

### 3. Run with auto-label + classification
```bash
python main.py --data mall_customers_full.csv --k 5 --autolabel --threshold 50
```

👉 Auto-label creates a `label` column from Spending Score (> threshold → 1).  
👉 To avoid leakage, Spending Score is dropped as a feature during classification when autolabel is used.  

---

## 📌 Example Output
```
Dataset loaded: (200, 5)
KMeans done with k = 5
Auto-label created: Score > 50 = 1

--- Logistic Regression ---
Accuracy: 1.0
ROC AUC: 1.0
Confusion:
 [[20  0]
 [ 0 20]]
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        20
           1       1.00      1.00      1.00        20

--- SVM ---
Accuracy: 1.0
ROC AUC: 1.0
Confusion:
 [[20  0]
 [ 0 20]]
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        20
           1       1.00      1.00      1.00        20
```

---

## 📝 Notes
- Perfect accuracy comes from synthetic labels (auto-label from Spending Score).  
- Real-world segmentation/classification would require more features and business-defined targets.  

---

✍️ **Author**: Nitin G Ghumare
