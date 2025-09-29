# 📌 Diabetes Detection - Case Study

## 🔹 Project Information
- **Author**      : Nitin G Ghumare  
- **Date**        : 28-Sep-2025  
- **Project Name**: Diabetes Detection  
- **Description** : A Machine Learning project using **Logistic Regression** and **SVM** to predict diabetes occurrence.  
  This project performs **EDA (Exploratory Data Analysis)**, evaluates the models with metrics, and saves important plots and results.

---

## 🔹 Dataset Description
- **Dataset Name** : DiabetesDataset.csv  
- **Target Column**: `Outcome` (0 = No Diabetes, 1 = Diabetes)  
- **Features**:
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  

**Example Records:**
| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
|-------------|---------|---------------|---------------|---------|------|--------------------------|-----|---------|
| 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                    | 50  | 1       |
| 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                    | 31  | 0       |
| 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                    | 32  | 1       |

---

## 🔹 Methodology
1. **Data Loading**: Load diabetes dataset.  
2. **Data Preprocessing**: Handle missing/invalid values (replace zeros with median), scale numeric features.  
3. **EDA & Visualization**: Create and save plots like outcome distribution, age histogram, glucose histogram, correlation heatmap.  
4. **Model Training**: Train Logistic Regression and Support Vector Machine (SVM).  
5. **Model Evaluation**: Evaluate using Accuracy, Confusion Matrix, and ROC AUC score.  
6. **Results Saving**: Save all generated plots and metrics in the `results/` folder.

---

## 🔹 Technologies Used
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  

---

## 🔹 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/NitinGhumare/machine-learning.git
   cd machine-learning/DiabetesDetection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python DiabetesPredictor.py
   ```

---

## 🔹 Sample Output
```
✅ Dataset loaded succesfully :
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
...

📏 Dimensions of dataset :  (768, 9)
📏 Dimensions after cleaning :  (768, 9)

🎯 Logistic Regression Accuracy :  0.7013
📊 Confusion matrix (LR) :
[[81 19]
 [27 27]]
📈 ROC AUC (LR) :  0.8128

🎯 SVM Accuracy :  0.7338
📊 Confusion matrix (SVM) :
[[84 16]
 [25 29]]
📈 ROC AUC (SVM) :  0.7963
```

---

## 🔹 Conclusion
- Logistic Regression achieved **~70% accuracy** with ROC AUC of **0.813**.  
- SVM performed slightly better with **~73% accuracy** and ROC AUC of **0.796**.  
- **Glucose, BMI, and Age** emerged as strong predictors of diabetes.  
- Demonstrates a **complete ML workflow**: preprocessing, visualization, model training, evaluation, and saving results.  

---

## 👨‍💻 Author
**Nitin G Ghumare**  
📧 Email: [nghumare570@gmail.com](mailto:nghumare570@gmail.com)  
🌐 GitHub: [NitinGhumare](https://github.com/NitinGhumare)  
