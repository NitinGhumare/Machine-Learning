# 📌 Titanic Survival Predictor - Case Study 2

## 🔹 Project Information
- **Author**      : Nitin G Ghumare  
- **Date**        : 28-Sep-2025  
- **Project Name**: Titanic Survival Predictor  
- **Description** : A Machine Learning project using **Logistic Regression** to predict whether a passenger survived or not on the Titanic.  
  This project performs **EDA (Exploratory Data Analysis)**, evaluates the model with metrics, and saves important plots and results.

---

## 🔹 Dataset Description
- **Dataset Name** : MarvellousTitanicDataset.csv  
- **Target Column**: `Survived` (0 = Not Survived, 1 = Survived)  
- **Features**:
  - PassengerId  
  - Age  
  - Fare  
  - Sex (0 = Male, 1 = Female)  
  - SibSp (Number of siblings/spouses aboard)  
  - Parch (Number of parents/children aboard)  
  - Pclass (1st, 2nd, 3rd Class)  
  - Embarked (Port of embarkation)  

**Example Records:**
| PassengerId | Age | Fare   | Sex | SibSp | Parch | Pclass | Embarked | Survived |
|-------------|-----|--------|-----|-------|-------|--------|----------|----------|
| 1           | 22  | 7.250  | 0   | 1     | 0     | 3      | 2        | 0        |
| 2           | 38  | 71.283 | 1   | 1     | 0     | 1      | 0        | 1        |
| 3           | 26  | 7.925  | 1   | 0     | 0     | 3      | 2        | 1        |

---

## 🔹 Methodology
1. **Data Loading**: Load Titanic dataset and remove unnecessary columns.  
2. **Data Preprocessing**: Handle missing values, encode categorical features, and scale numeric data.  
3. **EDA & Visualization**: Create and save plots like survival distribution, age histogram, fare histogram, correlation heatmap.  
4. **Model Training**: Train Logistic Regression on processed features.  
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
   cd machine-learning/TitanicPredictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python TitanicPredictor.py
   ```

---

## 🔹 Sample Output
```
✅ Dataset loaded succesfully :
   Passengerid   Age     Fare  Sex  SibSp  Parch  zero  Pclass  Embarked  Survived
0            1  22.0   7.2500    0      1      0     0       3       2.0         0
1            2  38.0  71.2833    1      1      0     0       1       0.0         1
...

📏 Dimensions of dataset :  (1309, 10)
📏 Dimensions after drop (if any) :  (1309, 8)
📏 Dimensions of Features :  (1309, 7)
📏 Dimensions of Labels :  (1309,)

🎯 Accuracy is :  0.7671
📊 Confusion matrix :
[[174  15]
 [ 46  27]]

📈 ROC AUC :  0.7860
```

---

## 🔹 Conclusion
- Logistic Regression achieved **~76% accuracy** on the Titanic dataset.  
- ROC AUC of **0.786** shows the model can reasonably discriminate survivors vs non-survivors.  
- EDA highlights **Gender** and **Passenger Class** as strong predictors of survival.  
- Demonstrates a **complete ML workflow**: preprocessing, visualization, model training, evaluation, and saving results.  

---

## 👨‍💻 Author
**Nitin G Ghumare**  
📧 Email: [nghumare570@gmail.com](mailto:nghumare570@gmail.com)  
🌐 GitHub: [NitinGhumare](https://github.com/NitinGhumare)  
