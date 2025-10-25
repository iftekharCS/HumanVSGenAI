# analyze.py
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tree_sitter_language_pack import get_parser

from data.loader import CodeDataset
from src.analysis.lexical import analyze_lexical_diversity


def batch_analyze(dataset: CodeDataset) -> pd.DataFrame:
    """Analyze all code samples and return results dataframe."""
    results = []
    parser = get_parser('python')
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Human
        human_raw = analyze_lexical_diversity(sample['human_code'], parser)
        human_metrics = {
            "total_tokens": human_raw["total_tokens"],
            "token_counts": human_raw["token_counts"],
            "zipf.slope": human_raw["zipf"]["slope"],
            "zipf.pairs": human_raw["zipf"]["rank_freq_pairs"],
            "heaps.b": human_raw["heaps"]["b"],
            "heaps.k": human_raw["heaps"]["k"],
            "heaps.pairs": human_raw["heaps"]["log_pairs"],
            "code_type": "human",
            "solution_num": i % 3 + 1
        }
        results.append(human_metrics)

        # AI
        ai_raw = analyze_lexical_diversity(sample['ai_code'], parser)
        ai_metrics = {
            "total_tokens": ai_raw["total_tokens"],
            "token_counts": ai_raw["token_counts"],
            "zipf.slope": ai_raw["zipf"]["slope"],
            "zipf.pairs": ai_raw["zipf"]["rank_freq_pairs"],
            "heaps.b": ai_raw["heaps"]["b"],
            "heaps.k": ai_raw["heaps"]["k"],
            "heaps.pairs": ai_raw["heaps"]["log_pairs"],
            "code_type": "ai",
            "solution_num": i % 3 + 1
        }
        results.append(ai_metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = df.groupby(['code_type', 'solution_num']).agg({
        'total_tokens': ['mean', 'std'],
        'zipf.slope': ['mean', 'std'],
        'heaps.b': ['mean', 'std'],
        'heaps.k': ['mean', 'std']
    })
    
    return df, summary

import matplotlib.pyplot as plt


def batch_plot_zipf_law(df: pd.DataFrame, max_samples_per_group: int = 10):
    """
    Plot log(rank) vs log(frequency) for multiple samples (AI and Human) in one plot.
    Uses precomputed log(rank), log(freq) pairs from analyze_zipf_law.
    Human curves are blue; AI curves are red.
    """
    plt.figure(figsize=(10, 7))

    color_map = {
        'human': 'blue',
        'ai': 'red'
    }

    for code_type in ['human', 'ai']:
        subset = df[df['code_type'] == code_type].head(max_samples_per_group)

        for idx, row in subset.iterrows():
            log_rank_freq = row['zipf.pairs']
            log_ranks, log_freqs = zip(*log_rank_freq)

            plt.plot(
                log_ranks,
                log_freqs,
                color=color_map[code_type],
                alpha=0.6,
                label=f"{code_type.capitalize()} #{row['solution_num']}"
            )

    plt.xlabel("log(Rank)")
    plt.ylabel("log(Frequency)")
    plt.title("Zipf's Law (Log-Log) Across Samples")
    plt.grid(True)
    plt.tight_layout()

    # Create a clean legend with only one entry per code_type
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Human'),
        Line2D([0], [0], color='red', lw=2, label='AI')
    ]
    plt.legend(handles=legend_elements)

    plt.show()


def batch_plot_heaps_law(df):
    """
    Log-log plot of Heaps' Law for multiple code samples.
    Each curve shows log(Vocabulary Size) vs. log(Token Count).
    """
    color_map = {"human": "blue", "ai": "red"}
    
    plt.figure(figsize=(10, 6))
    
    for idx, row in df.iterrows():
        code_type = row['code_type']
        growth_curve = row['heaps.pairs']
        
        if not growth_curve:
            continue
        
        token_counts, vocab_sizes = zip(*growth_curve)
        
        
        plt.plot(
            token_counts,
            vocab_sizes,
            color=color_map[code_type],
            alpha=0.6,
            label=f"{code_type.capitalize()} #{row['solution_num']}"
        )
    
    plt.title("Heaps' Law (Log-Log) Across Samples")
    plt.xlabel("log(Token Count)")
    plt.ylabel("log(Vocabulary Size)")
    
    # Avoid duplicated labels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='Human'),
        Line2D([0], [0], color='red', label='AI'),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_results(df: pd.DataFrame, summary: pd.DataFrame):
    """Save results to CSV files."""
    df.to_csv('results/detailed_results.csv', index=False)
    summary.to_csv('results/summary_statistics.csv')
    print("Results saved to detailed_results.csv and summary_statistics.csv")

if __name__ == "__main__":
    # Load dataset
    dataset = CodeDataset('data/dataset3.csv')
    
    # Run analysis
    detailed_results, summary_stats = batch_analyze(dataset)
    
    # Save results
    save_results(detailed_results, summary_stats)
    
    # Print quick summary
    print("\nSummary Statistics:")
    print(summary_stats)

    batch_plot_heaps_law(detailed_results)