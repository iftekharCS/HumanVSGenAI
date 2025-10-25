import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from data.loader import CodeDataset
from src.detection.detect import DetectCodeGPT

# Load model and tokenizer
model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize detector
detector = DetectCodeGPT(model=model, tokenizer=tokenizer)

# Load dataset
dataset = CodeDataset('data/dataset3.csv')

# Collect records: code, score, label
records = []
for item in dataset:
    # Human code
    human_code = item['human_code']
    _, human_score = detector.detect(human_code)
    records.append({'code': human_code, 'score': human_score, 'label': 0})

    # AI code
    ai_code = item['ai_code']
    _, ai_score = detector.detect(ai_code)
    records.append({'code': ai_code, 'score': ai_score, 'label': 1})

# Convert to DataFrame
df = pd.DataFrame(records)
df.to_csv('output_scores.csv', index=False)

# --- AUROC Calculation & Plotting ---
def calculate_auroc(df):
    thresholds = np.arange(0, 1.05, 0.05)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        df['pred'] = (df['score'] > thresh).astype(int)

        TP = ((df['pred'] == 1) & (df['label'] == 1)).sum()
        FP = ((df['pred'] == 1) & (df['label'] == 0)).sum()
        TN = ((df['pred'] == 0) & (df['label'] == 0)).sum()
        FN = ((df['pred'] == 0) & (df['label'] == 1)).sum()

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Sort for correct AUROC integration
    fpr_sorted, tpr_sorted = zip(*sorted(zip(fpr_list, tpr_list)))
    auroc = np.trapezoid(tpr_sorted, fpr_sorted)
    return auroc, tpr_sorted, fpr_sorted

# Calculate AUROC
auroc, tpr_sorted, fpr_sorted = calculate_auroc(df)
print(f"AUROC: {auroc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr_sorted, tpr_sorted, marker='o', label=f'AUROC = {auroc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
