from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_metrics(true_y: list[int] | np.ndarray[np.int32], probabilities: list[int] | np.ndarray[np.int32], info: str | None = None) -> dict[str, list[float] | list[str]]:
    """
    Calculates models accuracy, recall, sensitivity, precision, specificity and auc.
    The metrics are returned in a dict, so multiple outputs can be easily combined in a pd.DataFrame.

    :param true_y: A array or list of real answers.
    :param probabilities: A array or list of probabilities (probabilities lesser than 0.5 are considered 0 class).
    :param info: Optional. Additional info to be saved.
    :return: A dict with calculated metrics.
    """

    metrics = {}

    predictions = [i > 0.5 for i in probabilities]
    tn, fp, fn, tp = confusion_matrix(true_y, predictions).ravel()

    metrics["accuracy"] = [(tn + tp) / (tn + fp + fn + tp)]
    metrics["recall"] = [tp / (tp + fn)]
    metrics["sensitivity"] = [tn / (tn + fp)]
    metrics["precision"] = [tp / (tp + fp)]
    metrics["specificity"] = [tn / (tn + fp)]

    metrics["auc"] = [roc_auc_score(true_y, probabilities)]

    if info:
        metrics["info"] = [info]

    return metrics
