# 👤 Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to leave a telecom service, using demographic and service-related features.

---

## 🚀 Live Demo

**App Link**: https://pratyul-kr-c-c-p.streamlit.app/

---

## 🎯 Features

- 🔍 Selectable classifiers: Random Forest or Logistic Regression
- ⚙️ Adjustable threshold for probability-based classification
- 📊 Real-time prediction on user-provided input
- 📈 Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- 🗂️ Trained model & encoder persistence using Joblib
- 📉 Bar chart visualization of performance
- 🧪 Auto evaluation on test dataset

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** – ML algorithms & model evaluation
- **Pandas** – Data preprocessing
- **Streamlit** – Web interface
- **Joblib** – Model & encoder serialization
- **Matplotlib** – Metrics visualization

---

## 📊 Dataset

- **Source**: [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Target Column**: `Churn`
- **Important Features**: Contract, Tenure, Monthly Charges, Internet Service, etc.

---

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|:-------------------:|:--------:|:-----------:|:--------:|:----------:|
| Random Forest       | ✅        | ✅         | ✅      | ✅    |
| Logistic Regression | ✅        | ✅         | ✅      | ✅    |

➡️ Thresholds are tunable to balance between Recall and Precision.

---

## 🚧 Future Enhancements

- 🔁 Add more classifiers (e.g., XGBoost, SVM)
- 📊 Add feature importance visualizations
- 📝 Allow manual input of customer features via form

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/pratyul-kr/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python model_training.py
streamlit run main.py
```
