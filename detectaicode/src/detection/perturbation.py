import random
import numpy as np
from typing import List

def insert_spaces(code: str, alpha: float, lambda_spaces: int) -> str:
    """
    Insert random spaces into code.
    
    Args:
        code: Input code string
        alpha: Fraction of locations to perturb
        lambda_spaces: Poisson parameter for number of spaces
        
    Returns:
        Perturbed code string
    """
    # Find all possible insertion points (between tokens)
    tokens = code.split(' ')
    insertion_points = []
    
    # We'll insert spaces between existing tokens
    for i in range(1, len(tokens)):
        insertion_points.append(i)
    
    # Select subset of points to perturb
    num_to_perturb = int(alpha * len(insertion_points))
    points_to_perturb = random.sample(insertion_points, num_to_perturb)
    
    # Insert spaces at selected points
    perturbed_tokens = tokens.copy()
    for point in sorted(points_to_perturb, reverse=True):
        n_spaces = np.random.poisson(lambda_spaces) + 1  # At least one space
        perturbed_tokens.insert(point, ' ' * n_spaces)
    
    return ' '.join(perturbed_tokens)

def insert_newlines(code: str, beta: float, lambda_newlines: int) -> str:
    """
    Insert random newlines into code.
    
    Args:
        code: Input code string
        beta: Fraction of lines to perturb
        lambda_newlines: Poisson parameter for number of newlines
        
    Returns:
        Perturbed code string
    """
    lines = code.split('\n')
    num_to_perturb = int(beta * len(lines))
    lines_to_perturb = random.sample(range(len(lines)), num_to_perturb)
    
    perturbed_lines = []
    for i, line in enumerate(lines):
        perturbed_lines.append(line)
        if i in lines_to_perturb:
            n_newlines = np.random.poisson(lambda_newlines) + 1  # At least one
            perturbed_lines.extend([''] * n_newlines)
    
    return '\n'.join(perturbed_lines)

def perturb_code(code: str, alpha: float, beta: float, 
                lambda_spaces: int, lambda_newlines: int) -> str:
    """
    Apply random perturbation to code (either spaces or newlines).
    
    Args:
        code: Input code string
        alpha: Fraction of space locations to perturb
        beta: Fraction of newline locations to perturb
        lambda_spaces: Poisson parameter for space insertion
        lambda_newlines: Poisson parameter for newline insertion
        
    Returns:
        Perturbed code string
    """
    # Randomly choose perturbation type
    if random.random() < 0.5:
        return insert_spaces(code, alpha, lambda_spaces)
    else:
        return insert_newlines(code, beta, lambda_newlines)