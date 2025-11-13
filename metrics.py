"""
Evaluation metrics for B-Free
Includes: AUC, Accuracy, ECE (Expected Calibration Error), NLL (Negative Log-Likelihood)
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.special import softmax


def compute_auc(y_true, y_scores):
    """
    Compute Area Under the ROC Curve
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted probabilities for class 1
    
    Returns:
        auc: AUC score
    """
    return roc_auc_score(y_true, y_scores)


def compute_balanced_accuracy(y_true, y_pred):
    """
    Compute balanced accuracy (average of recall for each class)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        bacc: Balanced accuracy
    """
    # Calculate true positives and true negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Sensitivity (recall for class 1)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity (recall for class 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Balanced accuracy
    bacc = (sensitivity + specificity) / 2
    
    return bacc


def compute_ece(y_true, y_probs, n_bins=15):
    """
    Compute Expected Calibration Error (ECE)
    
    Measures the difference between predicted confidence and actual accuracy
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities (2D array: [N, 2] or 1D array for binary)
        n_bins: Number of bins for calibration
    
    Returns:
        ece: Expected calibration error
    """
    if len(y_probs.shape) == 2:
        # Take probability of positive class
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
    else:
        # Binary probabilities
        confidences = np.abs(y_probs - 0.5) * 2  # Convert to confidence
        predictions = (y_probs > 0.5).astype(int)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Calculate accuracy in this bin
            accuracy_in_bin = np.mean(predictions[in_bin] == y_true[in_bin])
            
            # Calculate average confidence in this bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # Add weighted difference to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_nll(y_true, y_probs, balanced=True):
    """
    Compute Negative Log-Likelihood
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities (2D array: [N, 2])
        balanced: If True, compute balanced NLL (class-weighted)
    
    Returns:
        nll: Negative log-likelihood
    """
    # Ensure y_probs is 2D
    if len(y_probs.shape) == 1:
        # Convert to 2D
        y_probs_2d = np.stack([1 - y_probs, y_probs], axis=1)
    else:
        y_probs_2d = y_probs
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-7
    y_probs_2d = np.clip(y_probs_2d, epsilon, 1 - epsilon)
    
    # Calculate log probabilities
    log_probs = np.log(y_probs_2d)
    
    if balanced:
        # Compute balanced NLL (average NLL per class)
        nll_per_class = []
        for class_idx in [0, 1]:
            mask = (y_true == class_idx)
            if np.sum(mask) > 0:
                class_nll = -np.mean(log_probs[mask, class_idx])
                nll_per_class.append(class_nll)
        
        nll = np.mean(nll_per_class)
    else:
        # Standard NLL
        nll = -np.mean(log_probs[np.arange(len(y_true)), y_true])
    
    return nll


def compute_all_metrics(y_true, y_probs, threshold=0.5, n_bins=15):
    """
    Compute all evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities for class 1 (or 2D array)
        threshold: Threshold for binary classification
        n_bins: Number of bins for ECE
    
    Returns:
        metrics: Dictionary of all metrics
    """
    # Convert to numpy if tensor
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_probs):
        y_probs = y_probs.cpu().numpy()
    
    # Get predictions
    if len(y_probs.shape) == 2:
        y_scores = y_probs[:, 1]  # Probability of class 1
        y_pred = (y_scores > threshold).astype(int)
    else:
        y_scores = y_probs
        y_pred = (y_probs > threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'auc': compute_auc(y_true, y_scores),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': compute_balanced_accuracy(y_true, y_pred),
        'ece': compute_ece(y_true, y_probs, n_bins=n_bins),
        'nll': compute_nll(y_true, y_probs, balanced=True)
    }
    
    return metrics


def print_metrics(metrics, prefix=""):
    """Pretty print metrics"""
    print(f"\n{prefix}Evaluation Metrics:")
    print("=" * 50)
    print(f"AUC:                {metrics['auc']:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"ECE:                {metrics['ece']:.4f}")
    print(f"NLL:                {metrics['nll']:.4f}")
    print("=" * 50)
