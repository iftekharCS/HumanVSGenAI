from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from ..analysis.logrank import calculate_log_rank_by_category

# from ..utils.parser import parse_code_tokens
from .perturbation import perturb_code


class DetectCodeGPT:
    def __init__(self, model, tokenizer, alpha=0.5, beta=0.5, lambda_spaces=3, lambda_newlines=2):
        """
        Initialize DetectCodeGPT detector.
        
        Args:
            model: The language model used for scoring
            alpha: Fraction of space locations to perturb
            beta: Fraction of newline locations to perturb
            lambda_spaces: Poisson parameter for space insertion
            lambda_newlines: Poisson parameter for newline insertion
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.beta = beta
        self.lambda_spaces = lambda_spaces
        self.lambda_newlines = lambda_newlines

    def calculate_npr_score(self, code: str, model, tokenizer, num_perturbations=20):
        category = "whitespace"
        orig_log_rank = calculate_log_rank_by_category(code, model, tokenizer, category)
        orig_log_rank = orig_log_rank.get(category)
        perturbed_log_ranks = []
        for _ in range(num_perturbations):
            perturbed = perturb_code(
                code,
                alpha=self.alpha,
                beta=self.beta,
                lambda_spaces=self.lambda_spaces,
                lambda_newlines=self.lambda_newlines
            )
            score = calculate_log_rank_by_category(perturbed, model, tokenizer, category)
            logrank_value = score.get(category)
            if logrank_value is not None:
                perturbed_log_ranks.append(logrank_value)

        mean_perturbed = np.mean(perturbed_log_ranks)
        return mean_perturbed / orig_log_rank
    
    def detect(self, code: str, num_perturbations: int = 10, threshold: float = None) -> Tuple[bool, float]:
        """
        Detect if code is machine-generated.
        
        Args:
            code: The code snippet to analyze
            num_perturbations: Number of perturbations to generate
            threshold: Decision threshold (if None, returns score)
            
        Returns:
            Tuple of (is_machine_generated, detection_score)
        """
        
        # Compute detection score (difference between original and perturbed)
        detection_score = self.calculate_npr_score(code, self.model, self.tokenizer, num_perturbations)
        
        # Apply threshold if provided
        if threshold is not None:
            return (detection_score > threshold, detection_score)
        return (None, detection_score)

    def calculate_auroc(self, scores, labels):
        # Convert to numpy arrays
        scores = np.array(scores)
        labels = np.array(labels)

        # Sort by descending scores
        desc_score_indices = np.argsort(-scores)
        scores = scores[desc_score_indices]
        labels = labels[desc_score_indices]

        # Total positives and negatives
        P = np.sum(labels == 1)
        N = np.sum(labels == 0)

        tpr_list = []
        fpr_list = []

        tp = 0
        fp = 0

        for i in range(len(scores)):
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / P if P > 0 else 0)
            fpr_list.append(fp / N if N > 0 else 0)

        # Sort by FPR for integration
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)

        # Trapezoidal integration of TPR over FPR
        auroc = np.trapezoid(tpr_array, fpr_array)
        return auroc