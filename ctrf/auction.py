import argparse
from collections import defaultdict
import numpy as np
import operator
import os
import pandas
import pickle
import time

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from .metrics import compute_auc, compute_model_auc, compute_model_metrics
from .models import train_rf, train_ctrf

def run_selection(seed, n_samples, auction_size, n_auctions):
    seed +=1 
    np.random.seed(seed)
    ind = np.random.randint(0, n_samples, size=auction_size*n_auctions)
    return ind, seed+1

def run_auction(dataset, seed, model, epsilon, auction_size, n_auctions, max_slate, base_welfare=None, welfare_inc=None):
    seed += 1
    np.random.seed(seed)

    name = dataset['name']
    x_te = dataset['X']
    y_te = dataset['y']
    ind = dataset['ind']
    auction_type = dataset['auction_type']
    reserve = dataset['reserve']

    # Copy our sampled data
    x, y = x_te[ind, :].copy(), y_te[ind].copy()

    # True PClicks, with some noise
    pclick = np.clip(model.predict_proba(x)[:, 1], .000001, 1-.000001)
    logit = np.log(pclick / (1-pclick))
    noisy_pclick = 1 / (1 + np.exp(- (logit + np.random.normal(scale=epsilon, size=len(ind)))))

    print(name, 'True AUC:', compute_auc(pclick, y), 'Noisy AUC:', compute_auc(noisy_pclick, y))

    # Build table with auctionid, dataid, and pclick
    df = pandas.DataFrame(zip(ind, logit, noisy_pclick), columns=['SampleId', 'Logit', 'PClick'])
    df['AuctionId'] = [int(i / auction_size) for i in range(auction_size*n_auctions)]

    # We have either a randomized dataset, or a greedy dataset.
    # Use this to create a temporary ranking pclick that's strictly for
    # ordering.  It won't be used in the training data.
    if auction_type == 'random':
        df['RankingPClick'] = np.random.permutation(df['PClick'])
    else:
        df['RankingPClick'] = df['PClick']

    df = df[df['RankingPClick'] >= reserve].reset_index(drop=True)

    # Rank by pclick, then get position and layout.  Apply max_slate here.
    df = df.sort_values(['AuctionId', 'RankingPClick'], ascending=False)
    df['Position'] = df.groupby('AuctionId').cumcount()
    df = df[df['Position'] < max_slate]

    # Random auctions pick a random layout
    if auction_type == 'random':
        m = np.max(df['AuctionId'])
        ids = np.arange(m)
        layouts = np.random.randint(1, max_slate+1, len(ids))
        layout_df = pandas.DataFrame(zip(ids, layouts), columns=['AuctionId', 'Layout'])
        df = df.join(layout_df, on='AuctionId', how='inner', lsuffix='', rsuffix='_dup')
        df.drop(list(df.filter(regex='_dup$')), axis=1, inplace=True)
        df = df[df['Position'] < df['Layout']]
    elif auction_type == 'welfare':
        df['Welfare'] = df.groupby('AuctionId')['PClick'].transform('cumsum') / (df['Position'] + 1)
        df['Welfare_Cut'] = base_welfare * (welfare_inc ** (df['Position']))
        df = df.query('Welfare > Welfare_Cut').copy()
        df['Layout'] = df.groupby('AuctionId')['AuctionId'].transform('count')

    else:
        df['Layout'] = df.groupby('AuctionId')['AuctionId'].transform('count')

    # print(df['BasePClick'] - df['TruePClick'])
    df['MeanLogit'] = df['Logit'].groupby(df['AuctionId']).transform('mean')
    df['TruePClick'] = 1 / (1 + np.exp(- (df['Logit'] + df['MeanLogit'])))

    # Rank by PClick, then cascade to generate clicks
    df['Uniform'] = np.random.uniform(size=len(df))
    df['WouldClick'] = np.where(df['Uniform'] <= df['TruePClick'], 1, 0)
    df['Click'] = 0
    df.loc[df["WouldClick"].ne(0).groupby(df['AuctionId']).idxmax(),'Click']=1
    df['Click'] = df['Click'] * df['WouldClick']

    df['AuctionPClick_CumMean'] = df.groupby('AuctionId')['PClick'].transform('cumsum') / (df['Position'] + 1) 
    
    df.drop(columns=['Uniform', 'WouldClick', 'RankingPClick', 'TruePClick'], inplace=True)

    print(df.groupby('Layout').AuctionId.nunique())

    return df, seed+1

def construct_auction_dataset(dataset, samples=None, position=None):
    # X = np.hstack((dataset['auctions'][['PClick', 'Position']], dataset['X'][dataset['auctions']['SampleId']]))
    if position:
        dataset = dataset.copy()
        dataset['auctions'] = dataset['auctions'][dataset['auctions']['Position'] == position]

    X = np.hstack((dataset['auctions'][['PClick', 'Position', 'Layout']], dataset['X'][dataset['auctions']['SampleId']]))
    # X = dataset['X'][dataset['auctions']['SampleId']]
    y = dataset['auctions']['Click']

    if samples:
        X = X[:samples, :]
        y = y[:samples]

    return X, y

def run_auction_evaluation(args):
    start = time.time()
    print('Arguments', args)
    X, y =  make_classification(  n_samples=args.n_tr_samples + args.n_te_cnt_samples + args.n_te_trt_samples + args.n_te_rnd_samples
                                , n_features=args.n_features
                                , n_informative=args.n_informative
                                , n_redundant=0
                                , n_clusters_per_class=args.n_clusters_per_class
                                , class_sep=args.class_sep
                                , random_state=args.seed)
    args.seed += 1

    print('Splitting Datasets')
    samples = np.array([0, args.n_tr_samples, args.n_te_cnt_samples, args.n_te_trt_samples, args.n_te_rnd_samples])
    datasets = {name : {'name':name, 'start':start, 'end':end, 'samples':samples, 'auction_type':auction_type, 'reserve':reserve} for name, start, end, samples, auction_type, reserve in zip(['oracle', 'cnt', 'trt', 'rnd'], np.cumsum(samples), np.cumsum(samples)[1:], samples[1:], [None, 'welfare', 'welfare', 'random'], [None, args.control_reserve, args.treatment_reserve, args.treatment_reserve])}

    for dataset, info in datasets.items():
        print(dataset, info['start'], info['end'], info['samples'])
        info['X'], info['y'] = X[info['start']:info['end'], :].copy() , y[info['start']:info['end']].copy()

    # Creates an oracle pclick that ignores position and simply observes c/nc
    oracle, args.seed = train_rf(datasets['oracle']['X'], datasets['oracle']['y'], seed=args.seed, n_estimators=args.oracle_n_estimators, min_samples_leaf=args.oracle_min_samples_leaf)
    print('Oracle Created -- Test AUC Control:', compute_model_auc(oracle, datasets['cnt']['X'], datasets['cnt']['y']))
    print('Oracle Created -- Test AUC Treatment:', compute_model_auc(oracle, datasets['trt']['X'], datasets['trt']['y']))

    # Run selection
    datasets['rnd']['ind'], args.seed = run_selection(args.seed, datasets['rnd']['samples'], args.control_auction_size, args.n_rnd_auction)
    datasets['cnt']['ind'], args.seed = run_selection(args.seed, datasets['cnt']['samples'], args.control_auction_size, args.n_auctions)
    datasets['trt']['ind'], args.seed = run_selection(args.seed, datasets['trt']['samples'], args.treatment_auction_size, args.n_auctions)

    # Run the auction
    datasets['rnd']['auctions'], args.seed = run_auction(datasets['rnd'], args.seed, oracle, args.epsilon, args.control_auction_size, args.n_rnd_auction, args.random_max_slate)
    datasets['cnt']['auctions'], args.seed = run_auction(datasets['cnt'], args.seed, oracle, args.epsilon, args.control_auction_size, args.n_auctions, args.control_max_slate, base_welfare=args.control_welfare, welfare_inc=args.control_welfare_inc)
    datasets['trt']['auctions'], args.seed = run_auction(datasets['trt'], args.seed, oracle, args.epsilon, args.treatment_auction_size, args.n_auctions, args.treatment_max_slate, base_welfare=args.treatment_welfare, welfare_inc=args.treatment_welfare_inc)

    models = defaultdict(dict)
    print('Train RF Models')
    models['rnd_rf']['model'], args.seed = train_rf(*construct_auction_dataset(datasets['rnd']), seed=args.seed, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)
    models['cnt_rf']['model'], args.seed = train_rf(*construct_auction_dataset(datasets['cnt']), seed=args.seed, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)
    models['cnt_rnd_ctrf_clo']['model'], args.seed = train_ctrf(*construct_auction_dataset(datasets['rnd']), *construct_auction_dataset(datasets['cnt']), seed=args.seed, addnotclobber=False, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)
    models['cnt_rnd_ctrf_add']['model'], args.seed = train_ctrf(*construct_auction_dataset(datasets['rnd']), *construct_auction_dataset(datasets['cnt']), seed=args.seed, addnotclobber=True, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)

    for key, data in models.items():
        data['metrics'] = compute_model_metrics(data['model'], *construct_auction_dataset(datasets['trt']))
        print(key, data['metrics'])

    print('Total Runtime', time.time() - start)

    if not os.path.exists(f'data_{args.timetag}'):
        os.makedirs(f'data_{args.timetag}')

    pickle.dump([args, datasets, models], open(f'data_{args.timetag}/data_{args.welfare_delta}_{args.seed}.pkl', 'wb'))

    # print('Eval rnd_rf on trt', compute_model_metrics(rnd_rf, *construct_auction_dataset(datasets['trt'])))
    # print('Eval cnt_rf on trt', compute_model_metrics(cnt_rf, *construct_auction_dataset(datasets['trt'])))
    # print('Eval cnt_cnd_ctrf_clo on trt', compute_model_metrics(cnt_cnt_ctrf_clo, *construct_auction_dataset(datasets['trt'])))
    # print('Eval cnt_rnd_ctrf_clo on trt', compute_model_metrics(cnt_rnd_ctrf_clo, *construct_auction_dataset(datasets['trt'])))
