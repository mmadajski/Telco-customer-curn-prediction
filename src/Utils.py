from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import pandas as pd


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


def get_model_metrics(model, model_name: str, X_train: pd.DataFrame, Y_train: list[int] | np.ndarray[np.int32], X_test: pd.DataFrame, Y_test: list[int] | np.ndarray[np.int32]) -> pd.DataFrame:
    """
    Creates a pd.DataFrame containing model metrics calculated on training and test data.

    :param model: Sklearn model.
    :param model_name: The name under which the model will be named inside the returned pd.DataFrame.
    :param X_train: Training data.
    :param Y_train: Training answers.
    :param X_test: Test data.
    :param Y_test: Test answers.
    :return: pd.DataFrame
    """
    probabilities_train = model.predict_proba(X_train)[:, 1]
    probabilities_test = model.predict_proba(X_test)[:, 1]
    metrics_train_tree = pd.DataFrame(calculate_metrics(Y_train, probabilities_train, model_name + "_train"))
    metrics_test_tree = pd.DataFrame(calculate_metrics(Y_test, probabilities_test, model_name + "_test"))

    return pd.concat([metrics_train_tree, metrics_test_tree])


def save_roc_curve_data(model, file_name: str, X_test: pd.DataFrame, Y_test: list[int] | np.ndarray[np.int32]) -> None:
    """
    Calculates false-positive and true-positive rates for models to be saved in a csv file.

    :param model: Sklearn model.
    :param file_name: The name under which the file is to be saved.
    :param X_test: Test data.
    :param Y_test: Test answers.
    """
    probabilities_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, probabilities_test)
    model_roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    model_roc_data.to_csv("..\\metrics\\" + file_name + ".csv", sep=";", decimal=",")
