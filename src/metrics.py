import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_audet(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the DET curve. Lower is better.

    DET plots FNR vs FPR. Since FNR = 1 - TPR, AuDET = 1 - AuROC.
    """
    auroc = roc_auc_score(labels, scores)
    return 1.0 - auroc


def compute_apcer_at_bpcer(
    labels: np.ndarray, scores: np.ndarray, bpcer_target: float = 0.01
) -> float:
    """APCER at a given BPCER operating point.

    From ROC curve: BPCER = FPR, APCER = 1 - TPR = FNR.
    We find the threshold where BPCER <= bpcer_target and report the best APCER.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    valid = fpr <= bpcer_target
    if not valid.any():
        return float(fnr[0])

    return float(fnr[valid].min())
