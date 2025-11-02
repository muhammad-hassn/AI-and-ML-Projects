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
<img width="419" height="295" alt="image" src="https://github.com/user-attachments/assets/ae9b047b-5cd9-4d50-bc49-91aad2b0d69f" />


### 🔹 Age Distribution by Survival
<img width="530" height="354" alt="image" src="https://github.com/user-attachments/assets/c7e62f1a-51eb-4d78-84ac-f3b64ce447d9" />


### 🔹 Passenger Class vs Survival
<img width="417" height="293" alt="image" src="https://github.com/user-attachments/assets/3fb88e19-5256-4c9e-a9a9-fd7ae95717a2" />


### 🔹 Confusion Matrix
<img width="340" height="295" alt="image" src="https://github.com/user-attachments/assets/c377c33d-60f0-46d4-a5b7-30f87b0921cd" />


### 🔹 ROC Curve
<img width="403" height="354" alt="image" src="https://github.com/user-attachments/assets/3ef694ba-79a9-4dc0-99a8-0f389958fa0e" />


### 🔹 Feature Importance
<img width="667" height="352" alt="image" src="https://github.com/user-attachments/assets/6020e960-a53e-4636-8a53-4675a403df13" />


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

