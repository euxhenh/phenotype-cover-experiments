from sklearn.metrics.cluster import pair_confusion_matrix
from itertools import combinations


def ari(labels_true, labels_pred):
    """
    Numerically stable adjusted rand index.
    """
    conf_matrix_ = pair_confusion_matrix(labels_true, labels_pred)
    (tn, fp), (fn, tp) = conf_matrix_/conf_matrix_.sum()
    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def iou(set1, set2):
    """
    Intersection over union score.
    """
    if len(set1) == 0 and len(set2) == 0:
        return 0
    set1 = set(set1)
    set2 = set(set2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


def stability(sets):
    """
    Stability score given a list of lists.
    """
    score = 0
    for (x, y) in combinations(range(len(sets)), 2):
        score += iou(sets[x], sets[y])
    score = 2 * score / (len(sets) * (len(sets) - 1))
    return score
