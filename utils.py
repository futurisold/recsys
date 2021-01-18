from sklearn.metrics import roc_curve, auc


def auc_score(scores):
    targets, preds = scores['targets'], scores['preds']
    fpr, tpr, _ = roc_curve(targets, preds)
    return auc(fpr, tpr)
