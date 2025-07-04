from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import pandas as pd
import numpy as np

df = pd.read_csv('D:\\Users\\user\\git\\study\\cote\\aiEngineerTest\\pathogenicity_scores.csv')

y_true = df['LABEL']

def find_best_threshold(y_true, y_score):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_thresh = 0
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    return best_thresh, best_f1

for name in ['SCORE_A', 'SCORE_B', 'SCORE_C']:
    y_score = df[name]
    
    y_pred = (y_score >= 0.5).astype(int)

    best_t, best_f1 = find_best_threshold(y_true, y_score)
    
    print(f'--- {name} ---')
    print(f'ROC-AUC:        {roc_auc_score(y_true, y_score):.4f}')
    print(f'PR-AUC:         {average_precision_score(y_true, y_score):.4f}')
    print(f'Accuracy:       {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision:      {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:         {recall_score(y_true, y_pred):.4f}')
    print(f'F1-score:       {f1_score(y_true, y_pred):.4f}')
    print(f"{name} 최적 threshold: {best_t:.3f} → 최대 F1-score: {best_f1:.4f}")
    print(f'ConfusionMatrix:\n{confusion_matrix(y_true, y_pred)}\n')