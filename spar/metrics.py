"""SPAR metrics for post-training target-label evaluation."""
import numpy as np

def classification_metrics_from_probabilities(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2:
        raise ValueError('probabilities must be [N, C]')
    if len(targets) != probabilities.shape[0]:
        raise ValueError('target count mismatch')
    if not np.isfinite(probabilities).all():
        raise ValueError('non-finite probabilities detected')
    probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    predictions = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(targets, predictions):
        confusion[int(true_label), int(pred_label)] += 1
    true_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    tp = np.diag(confusion)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp, dtype=np.float64), where=true_count > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp, dtype=np.float64), where=pred_count > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    valid = true_count > 0
    correct = int(tp.sum())
    return {
        'correct': correct,
        'ensemble_acc': float(100.0 * correct / max(len(targets), 1)),
        'macro_f1': float(f1[valid].mean()) if valid.any() else 0.0,
        'balanced_acc': float(recall[valid].mean()) if valid.any() else 0.0,
        'confusion_matrix': confusion.tolist(),
        'per_class_recall': recall.tolist(),
        'true_counts': true_count.tolist(),
        'pred_counts': pred_count.tolist(),
        'predictions': predictions.tolist(),
    }
