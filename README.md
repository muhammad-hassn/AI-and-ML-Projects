# AI-and-ML-Projects
## 🛳 Titanic Survival Prediction

## 🎯 Project Overview
This project predicts **whether a passenger survived or not** in the Titanic shipwreck disaster.  
It uses machine learning models to analyze features like passenger age, class, gender, and fare to determine survival probability.

---

## 📂 Dataset
We use the **Kaggle Titanic dataset**, which includes two CSV files:
- `train.csv` — contains labeled data with the `Survived` column (0 = Did not survive, 1 = Survived)
- `test.csv` — unlabeled data for final prediction

### Key Features:
| Feature | Description |
|----------|--------------|
| Pclass | Passenger class (1 = Upper, 2 = Middle, 3 = Lower) |
| Sex | Gender of the passenger |
| Age | Age in years |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
| Title | Extracted from Name (Mr, Miss, Mrs, etc.) |

---

## ⚙️ Workflow

1. **Load Data**
   - Import train and test CSV files into Google Colab.

2. **Exploratory Data Analysis (EDA)**
   - Check for missing values and overall data quality.

3. **Visualizations**
   - Explore how survival relates to gender, age, and class.

4. **Feature Engineering**
   - Extract new features like `Title` from names and handle missing values.

5. **Model Building**
   - Preprocessing using `ColumnTransformer` (Imputation + Encoding + Scaling)
   - Models used:
     - Random Forest Classifier 🌲
     - Logistic Regression 📈

6. **Evaluation**
   - Accuracy, Confusion Matrix, ROC Curve, and Feature Importance plots.

7. **Prediction & Export**
   - Predict survival for `test.csv` and export as `submission.csv`.

---

## 📊 Visualizations

### 🔹 Survival Count
<img width="432" height="296" alt="image" src="https://github.com/user-attachments/assets/66a34faa-a5fb-4eaa-a67b-2c0aa7ef7923" />


### 🔹 Survival by Gender
![Survival by Gender](plots/survival_by_gender.png)

### 🔹 Age Distribution by Survival
![Age Distribution](plots/age_distribution.png)

### 🔹 Passenger Class vs Survival
![Class vs Survival](plots/pclass_vs_survival.png)

### 🔹 Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

### 🔹 ROC Curve
![ROC Curve](plots/roc_curve.png)

### 🔹 Feature Importance
![Feature Importance](plots/feature_importance.png)

---

## 🧠 Model Performance (Random Forest)
| Metric | Score |
|---------|--------|
| Accuracy | ~0.83 |
| ROC AUC | ~0.88 |

---

## 💾 Output
- Trained model: `titanic_model.pkl`
- Submission file: `submission.csv`

---

## 🚀 Future Improvements
- Add advanced features (Family size, Deck extraction)
- Try XGBoost / LightGBM for better accuracy
- Hyperparameter optimization and ensemble models

---

**Author:** Muhammad Hassan  
**Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn  
**Environment:** Google Colab


## Business AI Meeting
## Image Caption AI

## Classifies Aircraft Damage

## K Mean
## Regression with Keras

