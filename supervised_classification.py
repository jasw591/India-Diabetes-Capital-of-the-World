import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("--- Starting Supervised Classification Pipeline ---")

# 1. Load data containing our unsupervised engineered feature
df = pd.read_csv('diabetes_with_clusters.csv')

# 2. Separate features (including Cluster) and targets
features_to_keep = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Cluster']

X = df[features_to_keep]
y = df['Outcome']

# 3. Stratified Data Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train CatBoost
print("Training CatBoost Classifier...")
cluster_col_idx = X_train.columns.get_loc('Cluster')
cat_model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_state=42, verbose=0)
cat_model.fit(X_train, y_train, cat_features=[cluster_col_idx])

# 5. Predictions and Accuracy Performance Reports
y_pred = cat_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n==========================================")
print(f"🎯 FINAL SUPERVISED MODEL ACCURACY: {accuracy:.2f}%")
print("==========================================\n")
print("📝 Detailed Performance Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Diabetic']))

# 6. Save final classification weights
cat_model.save_model('my_catboost.cbm')
print("CatBoost classification weights exported successfully!")
