import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

print("--- Starting Unsupervised Clustering Pipeline ---")

# 1. Load Data
df = pd.read_csv('diabetes.csv')

# 2. Select metrics for clustering
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_raw = df[features]

# 3. Preprocessing (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 4. Initialize and fit K-Means
print("Clustering data into 3 patient profile groups...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Evaluate Unsupervised Model Quality
sil_score = silhouette_score(X_scaled, df['Cluster'])
db_index = davies_bouldin_score(X_scaled, df['Cluster'])


print("     UNSUPERVISED MODEL METRICS RESULT     ")
print(f" Silhouette Score: {sil_score:.4f}")
print(f" Davies-Bouldin Index: {db_index:.4f}")
print("==========================================\n")

# 6. Save the cluster labels into a new dataset file and save modeling objects
df.to_csv('diabetes_with_clusters.csv', index=False)
joblib.dump(scaler, 'my_scaler.pkl')
joblib.dump(kmeans, 'my_kmeans.pkl')

print("Saved 'diabetes_with_clusters.csv' and model binaries successfully!")
