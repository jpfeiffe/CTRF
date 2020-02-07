import pandas
import numpy as np

from sklearn.ensemble import RandomForestClassifier

def train_rf(X, y, seed, **kwargs):
    seed += 1
    model = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=seed, **kwargs)
    model.fit(X, y)

    seed += 1
    return model, seed

def train_ctrf(X1, y1, X2, y2, seed, **kwargs):
    seed += 1
    model = RandomForestClassifier(criterion='entropy', random_state=seed, **kwargs)
    model.fit(X1, y1)

    for e in model.estimators_:
        df = pandas.DataFrame(zip(e.apply(X2), 1-y2, y2), columns=['LeafId', 'NoClick', 'Click'])
        df = df.groupby(['LeafId']).agg(NoClicks=pandas.NamedAgg(column='NoClick', aggfunc='sum'), Clicks=pandas.NamedAgg(column='Click', aggfunc='sum'))
        e.tree_.value[df.index.array] = np.expand_dims(df[['NoClicks', 'Clicks']].values, axis=1)

    seed += 1
    return model, seed