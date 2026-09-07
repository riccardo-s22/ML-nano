import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import umap  # Ensure umap-learn is installed

warnings.filterwarnings('ignore')

def load_data(desc_path, labels_path):
    desc = pd.read_csv(desc_path)
    labels = pd.read_csv(labels_path)
    labels['code_clean'] = labels['code'].str.replace('.', '_', regex=False)
    merged = desc.merge(labels[['code_clean', 'group']], left_on='code', right_on='code_clean')
    feature_cols = [c for c in desc.columns if c != 'code']
    X = merged[feature_cols].values.astype(np.float64)
    y = LabelEncoder().fit_transform(merged['group'].values)
    return X, y, feature_cols, merged['code'].values

# Parameters
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

desc_path = 'chirality_interp_descriptors.csv'
labels_path = 'sample_labels.csv'

X, y, feature_names, sample_codes = load_data(desc_path, labels_path)
print(f"Loaded {X.shape[0]} samples × {X.shape[1]} features")

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

y_true_all, y_pred_all, y_prob_all = [], [], []

print("Running UMAP + Random Forest Pipeline...")

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 1. Scale strictly within the fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. UMAP dimensionality reduction (n_neighbors must be small because N_train ~ 31)
    reducer = umap.UMAP(n_neighbors=5, n_components=3, min_dist=0.3, random_state=RANDOM_STATE + fold_idx)
    X_train_dr = reducer.fit_transform(X_train_scaled)
    X_test_dr = reducer.transform(X_test_scaled)  # UMAP supports transform!

    # 3. Train Random Forest on the UMAP components
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, class_weight='balanced')
    rf.fit(X_train_dr, y_train)

    y_prob = rf.predict_proba(X_test_dr)[:, 1]
    y_pred = rf.predict(X_test_dr)

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    y_prob_all.extend(y_prob)

# Aggregate
acc = accuracy_score(y_true_all, y_pred_all)
bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
auc = roc_auc_score(y_true_all, y_prob_all)

print("\n" + "="*50)
print("REPEATED CV RESULTS (UMAP + RANDOM FOREST)")
print("="*50)
print(f"Accuracy:          {acc:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"AUC:               {auc:.3f}")