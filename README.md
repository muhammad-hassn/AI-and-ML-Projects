# 🧠 AI-and-ML-Projects

---

## 🛳 Titanic Survival Prediction

### 🎯 Project Overview
This project predicts **whether a passenger survived or not** in the Titanic shipwreck disaster.  
It uses machine learning models to analyze features like passenger age, class, gender, and fare to determine survival probability.

---

### 📂 Dataset
We use the **Kaggle Titanic dataset**, which includes two CSV files:
- `train.csv` — contains labeled data with the `Survived` column (0 = Did not survive, 1 = Survived)
- `test.csv` — unlabeled data for final prediction

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

### ⚙️ Workflow
1. **Load Data** — Import CSV files into Google Colab  
2. **EDA** — Check for missing values and visualize data  
3. **Feature Engineering** — Handle missing values, extract titles  
4. **Model Building** — Random Forest & Logistic Regression  
5. **Evaluation** — Accuracy, Confusion Matrix, ROC, Feature Importance  
6. **Export Predictions** — Save final output as `submission.csv`

---

### 📊 Visualizations

#### 🔹 Survival Count
<img width="432" height="296" alt="Survival Count" src="https://github.com/user-attachments/assets/66a34faa-a5fb-4eaa-a67b-2c0aa7ef7923" />

#### 🔹 Survival by Gender
<img width="419" height="295" alt="Survival by Gender" src="https://github.com/user-attachments/assets/ae9b047b-5cd9-4d50-bc49-91aad2b0d69f" />

#### 🔹 Age Distribution by Survival
<img width="530" height="354" alt="Age Distribution" src="https://github.com/user-attachments/assets/c7e62f1a-51eb-4d78-84ac-f3b64ce447d9" />

#### 🔹 Passenger Class vs Survival
<img width="417" height="293" alt="Pclass vs Survival" src="https://github.com/user-attachments/assets/3fb88e19-5256-4c9e-a9a9-fd7ae95717a2" />

#### 🔹 Confusion Matrix
<img width="340" height="295" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/c377c33d-60f0-46d4-a5b7-30f87b0921cd" />

#### 🔹 ROC Curve
<img width="403" height="354" alt="ROC Curve" src="https://github.com/user-attachments/assets/3ef694ba-79a9-4dc0-99a8-0f389958fa0e" />

#### 🔹 Feature Importance
<img width="667" height="352" alt="Feature Importance" src="https://github.com/user-attachments/assets/6020e960-a53e-4636-8a53-4675a403df13" />

---

### 🧠 Model Performance (Random Forest)
| Metric | Score |
|---------|--------|
| Accuracy | ~0.83 |
| ROC AUC | ~0.88 |

---

### 💾 Output
- Trained model: `titanic_model.pkl`
- Submission file: `submission.csv`

---

### 🚀 Future Improvements
- Add new engineered features (Deck, Family size)
- Try XGBoost / LightGBM
- Hyperparameter tuning

---

**Author:** Muhammad Hassan  
**Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn  
**Environment:** Google Colab  

---

## 🌦️ Rainfall Prediction Classifier

### 📘 Overview
This project builds a **machine learning classifier** to predict whether it will **rain tomorrow** in Australia using historical weather data.  
It demonstrates the complete end-to-end ML pipeline — **data cleaning → feature engineering → model training → tuning → evaluation**.

---

### 📂 Dataset
Dataset: **`weatherAUS.csv`**  
Source: [Australian Bureau of Meteorology](http://www.bom.gov.au/)  
Kaggle: [Weather Dataset Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

| Column | Description | Example |
|:-------|:-------------|:--------|
| Date | Date of observation | 2016-05-10 |
| Location | City name | Sydney |
| MinTemp / MaxTemp | Daily min/max temperature (°C) | 10.5 / 22.3 |
| Rainfall | Rainfall amount (mm) | 5.4 |
| WindGustDir / WindGustSpeed | Direction & speed of strongest gust | NNW / 39 |
| Humidity3pm | Humidity at 3pm (%) | 55 |
| RainTomorrow | Target variable (Yes/No) | Yes |

---

### ⚙️ Technologies Used
- **Python 3**
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **scikit-learn** (pipelines, preprocessing, model selection)
- **GridSearchCV** for hyperparameter tuning

---

### 🚀 Project Workflow
1. **Data Exploration & Cleaning**
   - Handle missing values  
   - Drop high-missing columns: `Evaporation`, `Sunshine`, `Cloud9am`, `Cloud3pm`  
   - Impute numeric features (mean) and categorical features (mode)

2. **Feature Engineering**
   - Extract `Year`, `Month`, `Day` from `Date`  
   - Encode categorical variables with OneHotEncoder  
   - Scale numeric columns with StandardScaler  

3. **Model Building**
   - Create ML pipelines for:
     - Logistic Regression
     - Random Forest Classifier  

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrices and ROC Curves  

5. **Hyperparameter Tuning**
   - GridSearchCV with StratifiedKFold for Random Forest  
   - Parameters tuned:  
     - `n_estimators`  
     - `max_depth`  
     - `min_samples_split`

---

### 📊 Visualizations

#### 🔹 Confusion Matrix (Logistic Regression)
<img width="404" height="341" alt="image" src="https://github.com/user-attachments/assets/d46bb073-7fbb-497c-a0bc-e5d14a83e0d1" />

#### 🔹 Confusion Matrix (Random Forest)
<img width="405" height="343" alt="image" src="https://github.com/user-attachments/assets/68963e23-2fa1-4fad-b7df-a469e6547ea4" />

#### 🔹 Confusion Matrix (Random Forest)
<img width="402" height="343" alt="image" src="https://github.com/user-attachments/assets/791cc5fb-8fcf-4838-a859-105a0e6d683b" />

---

### 🧠 Results Summary
| Model | Accuracy | Notes |
|:------|:----------:|:------|
| Logistic Regression | ~83% | Good baseline model |
| Random Forest | ~86–88% | Strong non-linear model |
| Tuned Random Forest | **~89–90%** | Best final model |

**Conclusion:**  
The **Tuned Random Forest Classifier** achieved the best performance and is ideal for rainfall prediction.

---

### 💾 Output
- Trained model: `rainfall_model.pkl`  
- Predictions file: `rain_predictions.csv`

---

### 🧾 How to Run the Project
```python
# Load the dataset
df = pd.read_csv("weatherAUS.csv")
