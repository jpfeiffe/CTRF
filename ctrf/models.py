import pandas
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression
# from sklearn import svm

from copy import deepcopy

def train_rf(X, y, seed, **kwargs):
    start = time.time()
    seed += 1
    model = RandomForestClassifier(criterion='entropy', random_state=seed, **kwargs)
    model.fit(X, y)
    print('Runtime:', time.time()-start)
    seed += 1
    return model, seed

def train_combine_rf(X1, y1, X2, y2, seed, **kwargs):
    start = time.time()
    seed += 1
    model = RandomForestClassifier(criterion='entropy', random_state=seed, **kwargs)
    X_c = np.concatenate([X1, X2], axis=0)
    y_c = np.concatenate([y1, y2])
    model.fit(X_c, y_c)
    print('Runtime:', time.time()-start)
    seed += 1
    return model, seed

def train_ctrf(X1, y1, X2, y2, model, combine, seed,**kwargs):
    seed += 1
    # model = RandomForestClassifier(criterion='entropy', random_state=seed, **kwargs)
    # model.fit(X1, y1)
    #Copy an Oject
    start = time.time()
    model = deepcopy(model)
    for e in model.estimators_:
        if combine==0:
            df = pandas.DataFrame(zip(e.apply(X2), 1-y2, y2), columns=['LeafId', 'NoClick', 'Click'])
        else:
            X_c = np.concatenate([X1,X2],axis=0)
            y_c = np.concatenate([y1,y2])
            df = pandas.DataFrame(zip(e.apply(X_c), 1-y_c, y_c), columns=['LeafId', 'NoClick', 'Click'])
        df = df.groupby(['LeafId']).agg(NoClicks=pandas.NamedAgg(column='NoClick', aggfunc='sum'), Clicks=pandas.NamedAgg(column='Click', aggfunc='sum'))
        e.tree_.value[df.index.array] = np.expand_dims(df[['NoClicks', 'Clicks']].values, axis=1)
    print('Runtime:', time.time()-start)
    seed += 1
    return model, seed


def calculate_weight(train_X,testing_X):
    pool_X=np.vstack([train_X,testing_X])
    pool_Y=np.hstack([np.zeros(train_X.shape[0]),np.ones(testing_X.shape[0])])
    model=LogisticRegression(solver='liblinear')
    model.fit(pool_X,pool_Y)
    pred=model.predict_proba(train_X)
    weights=pred[:,1]/pred[:,0]
    weights=weights/np.mean(weights)
    return weights

def train_lr_model(X,y,seed,**kwargs):
    start = time.time()
    seed += 1
    model = LogisticRegression(solver='liblinear', random_state=seed)
    model.fit(X,y)
    print('Runtime:', time.time() - start)
    seed += 1
    return model,seed

def train_gbdt_model(X,y,seed,**kwargs):
    start = time.time()
    seed += 1
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X,y)
    print('Runtime:', time.time() - start)
    seed += 1
    return model,seed

def train_lr_weight_model(X,y,weights,seed,**kwargs):
    start = time.time()
    seed += 1
    model = LogisticRegression(solver='liblinear', random_state=seed)
    model.fit(X,y,sample_weight=weights)
    print('Runtime:', time.time() - start)
    seed += 1
    return model,seed

def train_gbdt_weight_model(X,y,weights,seed,**kwargs):
    start = time.time()
    seed += 1
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X,y,sample_weight=weights)
    print('Runtime:', time.time() - start)
    seed += 1
    return model,seed

