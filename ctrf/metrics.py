import operator
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,log_loss

###AUC Metric
def compute_auc(preds, ys):
    preds = sorted(zip(preds, ys), key=operator.itemgetter(0), reverse=True)
    pred_p_te, pred_y_te = zip(*preds)
    return roc_auc_score(pred_y_te, pred_p_te)

def compute_model_auc(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    return compute_auc(model.predict_proba(x_te)[:, 1], y_te)

###F1 Score
def compute_model_f1(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    return f1_score(model.predict(x_te), y_te)

###Bias
def compute_model_bias(model, x_te,y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    pred_ctr = np.mean(model.predict_proba(x_te)[:, 1])
    real_ctr = np.mean(y_te)
    return  abs(pred_ctr-real_ctr)/real_ctr

###RIG
def compute_model_rig(model, x_te, y_te):
    x_te = x_te.copy()
    y_te = y_te.copy()
    real_ctr = np.mean(y_te)
    real_entropy = - np.log(real_ctr)*real_ctr- np.log(1-real_ctr)*(1-real_ctr)
    pred_y = model.predict_proba(x_te)[:, 1]
    l_score = - log_loss(y_te, pred_y)

    return (real_entropy+l_score)/real_entropy

