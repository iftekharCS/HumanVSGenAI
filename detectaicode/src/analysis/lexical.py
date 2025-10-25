import math
from collections import Counter
from typing import Dict
from typing import List
from typing import Tuple

from tree_sitter import Node


def analyze_lexical_diversity(code: str, parser) -> Dict:
    """
    Analyze lexical diversity of code.
    
    Args:
        code: Input code string
        parser: Tree-sitter parser
        
    Returns:
        Dictionary with lexical diversity metrics
    """
    tree = parser.parse(bytes(code, "utf8"))
    tokens = collect_tokens(tree.root_node, code)
    
    # Token frequency analysis
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    
    # Zipf's law analysis
    zipf_results = analyze_zipf_law(token_counts)
    
    # Heaps' law analysis
    heaps_results = analyze_heaps_law(tokens)
    
    return {
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "zipf": zipf_results,
        "heaps": heaps_results
    }

def collect_tokens(node: Node, code: str) -> List[str]:
    """Recursively collect tokens from AST."""
    tokens = []
    code_bytes = code.encode("utf8")
    if node.child_count == 0:
        token = code_bytes[node.start_byte:node.end_byte].decode('utf8')
        tokens.append(token)
    else:
        for child in node.children:
            tokens.extend(collect_tokens(child, code))
    return tokens

def analyze_zipf_law(token_counts: Counter) -> Dict:
    """Analyze Zipf's law for token frequencies."""
    sorted_counts = sorted(token_counts.items(), key=lambda x: -x[1])
    ranks = range(1, len(sorted_counts)+1)
    frequencies = [count for _, count in sorted_counts]
    
    # Simple linear regression on log-log plot
    log_ranks = [math.log(r) for r in ranks]
    log_freqs = [math.log(f) for f in frequencies]
    
    # Calculate slope (should be close to -1 for Zipf's law)
    if len(log_ranks) > 1:
        slope = (len(log_ranks) * sum(lr*lf for lr, lf in zip(log_ranks, log_freqs)) - 
                sum(log_ranks) * sum(log_freqs)) / (
                len(log_ranks) * sum(lr**2 for lr in log_ranks) - 
                sum(log_ranks)**2)
    else:
        slope = 0
    
    return {
        "slope": slope,
        "rank_freq_pairs": list(zip(log_ranks, log_freqs))
    }

def analyze_heaps_law(tokens: List[str]) -> Dict:
    """Analyze Heaps' law for vocabulary growth."""
    vocabulary = set()
    growth_curve = []
    log_N = []
    log_V = []
    
    for i, token in enumerate(tokens):
        vocabulary.add(token)
        growth_curve.append((i+1, len(vocabulary)))  # (corpus_size, vocab_size)
    
    # Fit power law (V = k*N^b)
    if len(growth_curve) > 1:
        log_N = [math.log(n) for n, _ in growth_curve]
        log_V = [math.log(v) for _, v in growth_curve]
        
        b = (len(log_N) * sum(ln*lv for ln, lv in zip(log_N, log_V)) - 
             sum(log_N) * sum(log_V)) / (
             len(log_N) * sum(ln**2 for ln in log_N) - 
             sum(log_N)**2)
        k = math.exp((sum(log_V) - b * sum(log_N)) / len(log_N))
    else:
        b, k = 0, 0
    
    return {
        "b": b,
        "k": k,
        "growth_curve": growth_curve,
        "log_pairs": list(zip(log_N, log_V))
    }