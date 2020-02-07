import operator

from sklearn.metrics import roc_auc_score

def compute_auc(preds, ys):
    preds = sorted(zip(preds, ys), key=operator.itemgetter(0), reverse=True)
    pred_p_te, pred_y_te = zip(*preds)
    return roc_auc_score(pred_y_te, pred_p_te)

def compute_model_auc(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    return compute_auc(model.predict_proba(x_te)[:, 1], y_te)
