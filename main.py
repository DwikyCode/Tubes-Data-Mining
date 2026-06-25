import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

print("="*70)
print("KLASIFIKASI TINGKAT KESEJAHTERAAN WILAYAH DI INDONESIA")
print("MENGGUNAKAN LOGISTIC REGRESSION")
print("="*70)

# TAHAP 1: DATA UNDERSTANDING & VISUALISASI AWAL
print("\n" + "="*50)
print("TAHAP 1: DATA UNDERSTANDING")
print("="*50)

df = pd.read_csv("data/2021socio_economic_indonesia.csv")

print("\n5 Data Teratas:")
print(df.head())
print("\nDimensi Data:", df.shape)
print("\nInformasi Data:")
print(df.info())
print("\nDeskripsi Statistik:")
print(df.describe())

# Visualisasi Histogram Target Asli
plt.figure(figsize=(8,5))
sns.histplot(df["poorpeople_percentage"], kde=True)
plt.title("Distribusi Persentase Penduduk Miskin")
plt.show()

numerical_columns = [
    "poorpeople_percentage",
    "reg_gdp",
    "life_exp",
    "avg_schooltime",
    "exp_percap"
]

# Visualisasi Histogram Fitur
print("\nMenampilkan Histogram Fitur...")
for col in numerical_columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribusi {col}")
    plt.show()

# TAHAP 2: DATA PREPARATION & CLEANSING
print("\n" + "="*50)
print("TAHAP 2: DATA PREPARATION & CLEANSING")
print("="*50)

print("\n=== Missing Value ===")
print(df.isnull().sum())

print("\n=== Duplicate Data ===")
print(df.duplicated().sum())

# Eksekusi Data Cleansing Teks
df['province'] = df['province'].str.strip().str.title()
df['cities_reg'] = df['cities_reg'].str.strip().str.title()
print("Pembersihan spasi dan format huruf Title Case selesai dieksekusi")

# Visualisasi Boxplot untuk melihat Outlier
print("\nMenampilkan Boxplot...")
for col in numerical_columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot {col}")
    plt.show()

# Visualisasi Heatmap Korelasi
print("\nMenampilkan Heatmap Korelasi...")
plt.figure(figsize=(8,6))
sns.heatmap(df[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# TAHAP 3: PEMBUATAN TARGET BINER & SPLIT DATA
print("\n" + "="*50)
print("TAHAP 3: PEMBUATAN TARGET BINER & SPLIT DATA")
print("="*50)

# Rata-rata persentase penduduk miskin nasional BPS (Maret 2021)
THRESHOLD = 10.14
print(f"Ambang Batas (Threshold) Kemiskinan BPS 2021 = {THRESHOLD}%")

def kategori_kemiskinan(x):
    if x < THRESHOLD:
        return 0 # Kategori 0: Kemiskinan Rendah
    else:
        return 1 # Kategori 1: Kemiskinan Tinggi

df["kategori_kemiskinan"] = df["poorpeople_percentage"].apply(kategori_kemiskinan)

print("\nDistribusi Kelas Target Biner (0=Kemiskinan Rendah, 1=Kemiskinan Tinggi):")
print(df["kategori_kemiskinan"].value_counts())

features_lr = [
    "reg_gdp",
    "life_exp",
    "avg_schooltime",
    "exp_percap"
]

X = df[features_lr]
y = df["kategori_kemiskinan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== TRAIN TEST SPLIT ===")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# Standarisasi Data
scaler_lr = StandardScaler()
X_train_scaled = scaler_lr.fit_transform(X_train)
X_test_scaled = scaler_lr.transform(X_test)

print("\nData Train dan Test berhasil distandardisasi")

# TAHAP 4: MODELING & EVALUATION (LOGISTIC REGRESSION ONLY)
print("\n" + "="*50)
print("TAHAP 4: MODELING & EVALUATION")
print("="*50)

print("\n=== HYPERPARAMETER TUNING LOGISTIC REGRESSION ===")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'newton-cg']
}

base_lr = LogisticRegression(max_iter=5000, random_state=42)
grid_search = GridSearchCV(
    estimator=base_lr,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Proses Tuning Selesai")
print(f"Kombinasi Parameter Terbaik: {grid_search.best_params_}")
print(f"Akurasi Validasi (Cross-Val): {grid_search.best_score_:.4f}")

model_lr = grid_search.best_estimator_

print("\n=== PREDIKSI LOGISTIC REGRESSION ===")
y_pred = model_lr.predict(X_test_scaled)

print("\n=== EVALUASI LOGISTIC REGRESSION ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['0 (Kemiskinan Rendah)', '1 (Kemiskinan Tinggi)']))

print("\nConfusion Matrix Logistic Regression:")
print(confusion_matrix(y_test, y_pred))

# Visualisasi Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['0 (Kemiskinan Rendah)', '1 (Kemiskinan Tinggi)'])
plt.title("Confusion Matrix Logistic Regression")
plt.show()

print("\n=== KOEFISIEN PENGARUH VARIABEL X TERHADAP Y ===")
coef_df = pd.DataFrame(
    model_lr.coef_[0], 
    index=features_lr,
    columns=["Koefisien"]
)
print(coef_df)

# TAHAP 5: DEPLOYMENT PREPARATION
print("\n" + "="*50)
print("TAHAP 5: DEPLOYMENT PREPARATION")
print("="*50)

joblib.dump(model_lr, "logistic_regression.pkl")
joblib.dump(scaler_lr, "scaler.pkl")
print("Model Logistic Regression dan Scaler berhasil disimpan ke format .pkl")
print("Program Selesai")
print("="*70)