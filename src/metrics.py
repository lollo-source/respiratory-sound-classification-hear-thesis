from collections import OrderedDict


def confusion_matrix(y_true, y_pred, classes):
    position = {label: index for index, label in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for truth, prediction in zip(y_true, y_pred):
        if truth not in position or prediction not in position:
            raise ValueError("label outside explicit class order")
        matrix[position[truth]][position[prediction]] += 1
    return matrix


def classification_metrics(y_true, y_pred, classes):
    matrix = confusion_matrix(y_true, y_pred, classes)
    recalls = OrderedDict()
    f1_values = []
    correct = 0
    total = 0
    for index, label in enumerate(classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        recall = float(tp) / support if support else 0.0
        precision = float(tp) / predicted if predicted else 0.0
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        recalls[label] = recall
        f1_values.append(f1)
        correct += tp
        total += support
    return {
        "balanced_accuracy": sum(recalls.values()) / len(classes),
        "macro_f1": sum(f1_values) / len(classes),
        "accuracy": float(correct) / total if total else 0.0,
        "per_class_recall": recalls,
        "confusion_matrix": matrix,
    }

