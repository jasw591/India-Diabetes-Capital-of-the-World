# Unsupervised Risk Stratification: Coronary Heart Disease
An end-to-end data science and unsupervised machine learning pipeline designed to segment patient physiological risk profiles using the 4,240-patient Framingham Heart Study dataset. Without relying on historical target labels during training, this model successfully identifies high-risk clinical cohorts to optimize patient triaging.

# Project Overview & Deliverables
* Core Modeling: Built an unsupervised K-Means pipeline to cluster patients purely on continuous baseline physiological features (Age, BMI, Blood Pressure, Smoking Habits).
* Mathematical Optimization: Executed multi-K evaluation loops to establish peak cluster validation at K=2 using Silhouette Coefficient analysis, rejecting over-segmented micro-clusters (K=10) to prevent overfitting.
* Clinical Utility: Achieved a 68.56% corrected classification accuracy when mapped against historical 10-year Coronary Heart Disease (CHD) endpoints.
* Targeted Recall: Successfully isolated a high-risk cohort capturing 52% of all true medical events (Recall) completely unassisted by target labels.

# Tech Stack & Methodology
1. Language: Python
2. Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
3. Data Pipeline: 
  1. Missing values handled via robust Median Imputation.
  2. Distance-based metrics stabilized using StandardScaler normalization.
  3. Feature clustering via optimized KMeans
  4. Post-hoc label alignment mapping cluster profiles directly to clinical ground truth.

---

# Performance Summary
 Classification Report
```text
              precision    recall  f1-score   support

 Healthy (0)       0.89      0.72      0.79      3596
     CHD (1)       0.25      0.52      0.33       644

    accuracy                           0.69      4240
   macro avg       0.57      0.62      0.56      4240
weighted avg       0.79      0.69      0.72      4240
