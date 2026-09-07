import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

# --- LOAD YOUR RESULTS ---
# We load the file you just generated
df = pd.read_csv("batch_predictions_v12.csv")

# Filter only rows where we have ground truth
df = df.dropna(subset=['true_label'])

y_true = np.array([1 if x == "ALS" else 0 for x in df['true_label']])
y_prob = df['prob_als'].values

print(f"Loaded {len(df)} samples.")

# --- SEARCH FOR OPTIMAL THRESHOLD ---
thresholds = np.arange(0.01, 1.0, 0.01)
best_acc = 0
best_f1 = 0
best_thr_acc = 0.5
best_thr_f1 = 0.5

history = []

for thr in thresholds:
    y_pred = (y_prob >= thr).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_thr_acc = thr
        
    if f1 > best_f1:
        best_f1 = f1
        best_thr_f1 = thr
        
    history.append({'thr': thr, 'acc': acc, 'f1': f1})

print("\n--- OPTIMIZATION RESULTS ---")
print(f"Default (0.50) -> Accuracy: {accuracy_score(y_true, (y_prob>=0.5).astype(int)):.4f}")
print(f"Best Accuracy  -> {best_acc:.4f} at Threshold {best_thr_acc:.2f}")
print(f"Best F1 Score  -> {best_f1:.4f} at Threshold {best_thr_f1:.2f}")

# --- RE-EVALUATE WITH BEST THRESHOLD ---
final_thr = best_thr_f1
y_pred_new = (y_prob >= final_thr).astype(int)

print(f"\n--- NEW METRICS (Thr={final_thr:.2f}) ---")
print(confusion_matrix(y_true, y_pred_new))
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
print(f"Specificity: {tn / (tn+fp):.4f}")
print(f"Sensitivity: {tp / (tp+fn):.4f}")

# --- PLOT ---
plt.figure(figsize=(10, 5))
plt.plot([h['thr'] for h in history], [h['acc'] for h in history], label='Accuracy')
plt.plot([h['thr'] for h in history], [h['f1'] for h in history], label='F1 Score')
plt.axvline(x=final_thr, color='r', linestyle='--', label=f'Optimal ({final_thr:.2f})')
plt.title("Performance vs. Decision Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("threshold_calibration.png")
print("\nPlot saved to threshold_calibration.png")