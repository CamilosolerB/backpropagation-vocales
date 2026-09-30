"""Métricas de evaluación multiclase (matriz de confusión, precisión,
sensibilidad/recall, F1 y exactitud) calculadas solo con tipos nativos."""


def build_confusion_matrix(y_true, y_pred, n_classes):
    """Matriz de confusión: filas = clase real, columnas = clase predicha."""
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def accuracy(cm):
    total = sum(sum(row) for row in cm)
    if total == 0:
        return 0.0
    hits = sum(cm[i][i] for i in range(len(cm)))
    return hits / total


def per_class_metrics(cm):
    """Por cada clase: precisión, sensibilidad (recall) y F1."""
    n = len(cm)
    metrics = []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n)) - tp
        fn = sum(cm[c][s] for s in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        metrics.append(
            {"precision": precision, "recall": recall, "f1": f1, "tp": tp}
        )
    return metrics