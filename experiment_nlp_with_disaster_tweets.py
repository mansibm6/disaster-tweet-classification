"""Reusable experiment functions extracted from nlp_with_disaster_tweets.ipynb."""

from __future__ import annotations


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
    -----
    y_true = true labels in the form of a 1D array
    y_pred = predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1,
    }
    return model_results


def compare_baseline_to_new_results(baseline_results, new_model_results):
    for key, value in baseline_results.items():
        print(
            f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, Difference: {new_model_results[key] - value:.2f}"
        )


def build_model_6_cell_113(sentence_encoder_layer):
    """Build the original architecture without training or accessing data."""
    from tensorflow.keras import layers
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            sentence_encoder_layer,
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="model_6_USE",
    )
