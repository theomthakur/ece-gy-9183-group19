from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score

def average_precision_at_05(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = auc(recall, precision)
    return ap

def auc_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr)

def auc_pr(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)