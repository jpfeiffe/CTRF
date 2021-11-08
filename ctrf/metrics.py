import operator
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_abs_err(preds, ys):
    preds = sorted(zip(preds, ys), key=operator.itemgetter(0), reverse=True)
    pred_p_te, pred_y_te = zip(*preds)
    return np.mean(np.abs(np.array(pred_p_te) - np.array(pred_y_te)))

def compute_auc(preds, ys):
    preds = sorted(zip(preds, ys), key=operator.itemgetter(0), reverse=True)
    pred_p_te, pred_y_te = zip(*preds)
    return roc_auc_score(pred_y_te, pred_p_te)

def compute_model_auc(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    return compute_auc(model.predict_proba(x_te)[:, 1], y_te)

def compute_model_abs_err(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    return compute_abs_err(model.predict_proba(x_te)[:, 1], y_te)

def compute_model_metrics(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    abs_err = compute_abs_err(model.predict_proba(x_te)[:, 1], y_te)
    auc = compute_auc(model.predict_proba(x_te)[:, 1], y_te)
    return {'AbsErr' : abs_err, 'AUC' : auc}