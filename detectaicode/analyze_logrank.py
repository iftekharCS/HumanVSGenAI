import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from data.loader import CodeDataset
from src.analysis.logrank import calculate_log_rank
from src.analysis.logrank import calculate_log_rank_by_category


def batch_naturalness_analysis(dataset: CodeDataset, model, tokenizer):
    """
    Analyze log rank scores for all code samples in the dataset.
    
    Args:
        dataset: CodeDataset object
        model: Pretrained language model
        tokenizer: Corresponding tokenizer
        
    Returns:
        DataFrame with log rank scores
    """
    results = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Human code
        lr_human = calculate_log_rank(sample['human_code'], model, tokenizer)
        results.append({
            'code_type': 'human',
            'solution_num': i % 3 + 1,
            'logrank': lr_human
        })

        # AI code
        lr_ai = calculate_log_rank(sample['ai_code'], model, tokenizer)
        results.append({
            'code_type': 'ai',
            'solution_num': i % 3 + 1,
            'logrank': lr_ai
        })

    df = pd.DataFrame(results)
    
    # Group and summarize
    summary = df.groupby(['code_type', 'solution_num']).agg({
        'logrank': ['mean', 'std']
    })
    
    return df, summary

def batch_analyze(dataset: CodeDataset, model, tokenizer):
    """
    Analyze log rank scores by category for all code samples.
    """
    all_results = []

    for i in range(len(dataset)):
        sample = dataset[i]
        for code_type in ['human_code', 'ai_code']:
            cat_scores = calculate_log_rank_by_category(sample[code_type], model, tokenizer)
            for cat, score in cat_scores.items():
                all_results.append({
                    'code_type': 'human' if code_type == 'human_code' else 'ai',
                    'solution_num': i % 3 + 1,
                    'category': cat,
                    'logrank': score
                })

    df = pd.DataFrame(all_results)
    
    # Group and summarize
    summary = df.groupby(['code_type', 'category']).agg({
        'logrank': ['mean', 'std']
    }).reset_index()

    return df, summary

# Load tokenizer and model
model_name = "Salesforce/codegen-350M-mono"  # or any other causal model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

dataset = CodeDataset('data/dataset1.csv')

lr_detailed, lr_summary = batch_analyze(dataset, model, tokenizer)

labels = (lr_detailed['code_type'] == 'ai').astype(int).tolist()
scores = lr_detailed['logrank'].tolist()

# Save and display
lr_detailed.to_csv('results/logrank_detailed_results.csv', index=False)
lr_summary.to_csv('results/logrank_summary_statistics.csv')
print("\nLog Rank Statistics:")
print(lr_summary)
