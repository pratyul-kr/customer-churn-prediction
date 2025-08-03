import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.stats import randint

# Load Dataset
df = pd.read_csv("customer-churn.csv")

# Select Important Columns
df = df[['Contract', 'tenure', 'MonthlyCharges', 'InternetService', 'Churn']]
df.dropna(inplace=True)

# Encode Categorical Features
le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_churn = LabelEncoder()

df['Contract'] = le_contract.fit_transform(df['Contract'])
df['InternetService'] = le_internet.fit_transform(df['InternetService'])
df['Churn'] = le_churn.fit_transform(df['Churn'])  # Yes=1, No=0

# Features & Target
X = df[['Contract', 'tenure', 'MonthlyCharges', 'InternetService']]
y = df['Churn']

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True],
    'max_features': ['sqrt', 'log2']
}
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=15,
    cv=5,
    scoring='recall',
    verbose=1,
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
rf_best = random_search.best_estimator_

# 2. Logistic Regression
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
logreg.fit(X_train, y_train)

# Define label names
target_names = le_churn.inverse_transform([0, 1])  # ['No', 'Yes']

# Evaluation Function
def evaluate_model(model, name, X=X_test, y=y_test):
    y_pred = model.predict(X)
    print(f"\n====== {name} Performance ======")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=target_names))

# Evaluate Default Threshold Models
evaluate_model(rf_best, "Random Forest (Best Params)")
evaluate_model(logreg, "Logistic Regression")

# Threshold Tuning

# For Random Forest
y_proba_rf = rf_best.predict_proba(X_test)[:, 1]
y_pred_thresh_rf = (y_proba_rf >= 0.4).astype(int)
print("\n==== Random Forest with Tuned Threshold (0.4) ====")
print("Accuracy:", accuracy_score(y_test, y_pred_thresh_rf))
print("Precision:", precision_score(y_test, y_pred_thresh_rf))
print("Recall:", recall_score(y_test, y_pred_thresh_rf))
print("F1 Score:", f1_score(y_test, y_pred_thresh_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_thresh_rf, target_names=target_names))

# For Logistic Regression
y_proba_logreg = logreg.predict_proba(X_test)[:, 1]
y_pred_thresh_logreg = (y_proba_logreg >= 0.4).astype(int)
print("\n==== Logistic Regression with Tuned Threshold (0.4) ====")
print("Accuracy:", accuracy_score(y_test, y_pred_thresh_logreg))
print("Precision:", precision_score(y_test, y_pred_thresh_logreg))
print("Recall:", recall_score(y_test, y_pred_thresh_logreg))
print("F1 Score:", f1_score(y_test, y_pred_thresh_logreg))
print("Classification Report:")
print(classification_report(y_test, y_pred_thresh_logreg, target_names=target_names))

# Save Models & Encoders
joblib.dump(rf_best, "random_forest_model.pkl")
joblib.dump(logreg, "logistic_model.pkl")
joblib.dump(le_contract, "contract_encoder.pkl")
joblib.dump(le_internet, "internet_encoder.pkl")
joblib.dump(le_churn, "churn_encoder.pkl")

# Save Test Set
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)