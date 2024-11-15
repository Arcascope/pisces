"""Based on code originally written by Eric Canton and Franco Tavella, copyright 2024 Arcascope, Inc."""

import numpy as np
from sklearn.utils import compute_sample_weight
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


def wldm_to_sw_proba(wldm_vec: np.ndarray) -> np.ndarray:
    """Converts a WLDM vector to a sleep/wake probability vector, by taking  
        Pr(epoch[i] == sleep) := 1 - Pr(epoch[i] == wake)
        
        ."""
    return 1 - wldm_vec[:, 0]

def sw_accuracy(y_true, y_prob):
    """Sleep/Wake Accuracy (SWA) metric."""
    # Remove masked values
    selector = y_true >= 0
    y_true_filtered = y_true[selector]
    y_prob_filtered = y_prob[selector]
    # Convert to sw prediction
    y_true_filtered = np.array([
        y if y == 0 else 1 for y in y_true_filtered
    ])
    y_pred_filtered = np.argmax(y_prob_filtered, axis=1)
    y_pred_filtered = np.array([
        y if y == 0 else 1 for y in y_pred_filtered
    ])
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_true_filtered)
    sw_accuracy_value = accuracy_score(
        y_true_filtered,
        y_pred_filtered,
        sample_weight=sample_weights)
    return sw_accuracy_value


def get_sleep_probability(y_pred_prob, convert_from_wldm):
    """Get the sleep probability from the prediction."""
    if convert_from_wldm:
        sleep_probability = wldm_to_sw_proba(y_pred_prob)
    else:
        sleep_probability = y_pred_prob[:, 1]
    return sleep_probability


def wasa(y_true, y_pred_prob, sleep_accuracy=0.93,
         convert_from_wldm=True):
    """Wake Accuracy with Sleep Accuracy (WASA) metric."""
    # Convert to sleep/wake
    # y_true = wrap_labels(y_true, 8)
    selector = y_true >= 0 # remove masked values
    y_true_filtered = y_true[selector]
    y_true_filtered = np.array([
        y if y == 0 else 1 for y in y_true_filtered
    ])
    sleep_probability = get_sleep_probability(y_pred_prob, convert_from_wldm)[selector]

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_true_filtered)
    _, tpr, threshold = roc_curve(y_true_filtered, sleep_probability, pos_label=1,
                                    sample_weight=sample_weights)
    wasa_threshold = threshold[np.sum(tpr <= sleep_accuracy)]
    y_guess = sleep_probability > wasa_threshold
    guess_right = y_guess == y_true_filtered
    y_wake = y_true_filtered == 0
    n_wake = np.sum(y_wake)
    wake_accuracy = np.sum(y_wake * guess_right) / n_wake

    return wake_accuracy


def auroc(y_true, y_pred_prob, convert_from_wldm=True):
    """Area Under the ROC (AUC) metric."""
    selector = y_true >= 0 # remove masked values
    y_true_filtered = y_true[selector]
    y_true_filtered = np.array([
        y if y == 0 else 1 for y in y_true_filtered
    ])
    sleep_probability = get_sleep_probability(y_pred_prob, convert_from_wldm)[selector]
    auc_score_value = roc_auc_score(y_true_filtered, sleep_probability)
    return auc_score_value


def calculate_metrics_from_splits(preprocessed_data, models, splits):
    """Once we have the trained models and the splits, we can calculate the metrics."""
    metrics = {
        'sw_accuracy': [],
        'wasa': [],
        'auc': []
    }
    convert_from_wldm = False
    for model_idx, model in enumerate(models):
        # print(f"Model {model_idx}")
        for idx, test_idxs in enumerate(splits):
            # print(f"Split {idx}, test_idxs {test_idxs}")
            for test_idx in test_idxs[1]:
                # print(f"Test idx {test_idx}")
                X_test, y_true = preprocessed_data[test_idx]
                if X_test is None or y_true is None:
                    print(f"Skipping test_idx {test_idx} due to missing data")
                    print(f"X_test: {X_test}")
                    print(f"y_true: {y_true}")
                    continue
                y_prob = model.predict_probabilities(X_test)
                sw_accuracy_value = sw_accuracy(y_true, y_prob)
                wasa_value = wasa(y_true, y_prob, convert_from_wldm=convert_from_wldm)
                auc_value = auroc(y_true, y_prob, convert_from_wldm=convert_from_wldm)
                metrics['sw_accuracy'].append(sw_accuracy_value)
                metrics['wasa'].append(wasa_value)
                metrics['auc'].append(auc_value)
    return metrics