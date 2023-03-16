from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix
from _collections import OrderedDict
import numpy as np

def evaluate(y_true, y_pred, digits=4, cutoff='auto'):
    '''
    calculate several metrics of predictions

    Args:
        y_true: list, labels
        y_pred: list, predictions
        digits: The number of decimals to use when rounding the number. Default is 4
        cutoff: float or 'auto'

    Returns:
        evaluation: dict

    '''

    if cutoff == 'auto':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden = tpr-fpr
        cutoff = thresholds[np.argmax(youden)]

    y_pred_t = [1 if i > cutoff else 0 for i in y_pred]

    evaluation = OrderedDict()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
    evaluation['auc'] = round(roc_auc_score(y_true, y_pred), digits)
    evaluation['acc'] = round(accuracy_score(y_true, y_pred_t), digits)
    evaluation['sensitivity'] = round(recall_score(y_true, y_pred_t), digits)
    evaluation['specificity'] = round(tn / (tn + fp), digits)
    evaluation['F1 score'] = round(f1_score(y_true, y_pred_t), digits)
    evaluation['gmean'] = round(np.sqrt(recall_score(y_true, y_pred_t) * tn / (tn + fp)), digits)
    evaluation['cutoff'] = cutoff

    return evaluation


def Find_Optimal_Cutoff(TPR, FPR, threshold):

    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
