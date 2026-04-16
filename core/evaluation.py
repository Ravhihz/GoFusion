import numpy as np
from collections import Counter

def calculate_metrics_multiclass(y_true, y_pred):
    """
    Full-scratch evaluation (no sklearn)
    Returns metrics CONSISTENT dengan EvaluationMetrics model
    """

    labels = list(set(y_true))
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    per_class_metrics = {}
    precision_list = []
    recall_list = []
    f1_list = []

    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        per_class_metrics[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support
        }

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1_score": float(np.mean(f1_list)),
        "per_class": per_class_metrics  # ← TAMBAH INI
    }