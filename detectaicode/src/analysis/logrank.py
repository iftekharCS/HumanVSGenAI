import keyword
import tokenize
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch


def get_token_categories(code: str):
    """
    Returns a list of (token_str, category) pairs.
    """
    categories = []
    try:
        tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
        
        for token in tokens:
            tok_type = token.type
            tok_str = token.string

            if tok_type == tokenize.COMMENT:
                categories.append((tok_str, "comment"))
            elif tok_type == tokenize.NAME:
                if tok_str in keyword.kwlist:
                    categories.append((tok_str, "keyword"))
                else:
                    categories.append((tok_str, "identifier"))
            elif tok_type == tokenize.STRING or tok_type == tokenize.NUMBER:
                categories.append((tok_str, "literal"))
            elif tok_type == tokenize.OP:
                categories.append((tok_str, "operator"))
            elif tok_type == tokenize.NL or tok_type == tokenize.NEWLINE or tok_type == tokenize.INDENT or tok_type == tokenize.DEDENT:
                categories.append((tok_str, "whitespace"))
            elif tok_type == tokenize.ENCODING or tok_type == tokenize.ENDMARKER:
                continue
            else:
                categories.append((tok_str, "symbol"))
        return categories
    except IndentationError:
        return []


def calculate_log_rank_by_category(code: str, model, tokenizer, category=None) -> dict:
    inputs = tokenizer(code, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    offset_mapping = inputs['offset_mapping'][0]  # (token_count, 2)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Get token spans to map to categories
    code_token_categories = get_token_categories(code)

    def get_category(char_pos):
        for token_str, cat in code_token_categories:
            if code.find(token_str, char_pos) == char_pos:
                return cat
        return "unknown"

    category_log_ranks = defaultdict(list)

    for i in range(input_ids.size(1)):
        token_id = input_ids[0, i]
        rank = (sorted_indices[0, i] == token_id).nonzero().item() + 1
        log_rank = torch.log(torch.tensor(rank, dtype=torch.float)).item()

        # Map to character span, then to category
        start, end = offset_mapping[i].tolist()
        cat = get_category(start)
        category_log_ranks[cat].append(log_rank)

    # Compute average log rank per category
    avg_log_ranks = {cat: sum(vals) / len(vals) for cat, vals in category_log_ranks.items() if vals}

    if category is not None:
        return {category: avg_log_ranks.get(category, None)}
    return avg_log_ranks



def calculate_log_rank(code: str, model, tokenizer) -> float:
    """
    Calculate log rank for the code.
    """
    inputs = tokenizer(code, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    # Probabilities and rank calculations
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    ranks = torch.zeros_like(input_ids)
    for i in range(input_ids.size(1)):
        token_id = input_ids[0, i]
        rank = (sorted_indices[0, i] == token_id).nonzero().item()
        ranks[0, i] = rank + 1  # 1-based rank

    log_ranks = torch.log(ranks.float())
    avg_log_rank = log_ranks.mean().item()

    return avg_log_rank
