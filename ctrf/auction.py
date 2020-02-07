import argparse
import numpy as np
import operator
import pandas

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from .metrics import compute_auc, compute_model_auc

def run_selection(seed, n_samples, auction_size, n_auctions):
    seed +=1 
    np.random.seed(seed)
    ind = np.random.randint(0, n_samples, size=auction_size*n_auctions)
    return ind, seed+1

def run_auction(dataset, seed, model, epsilon, auction_size, n_auctions, max_slate):
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
    pclick = model.predict_proba(x)[:, 1]
    logit = np.log(pclick / (1-pclick))
    noisy_pclick = 1 / (1 + np.exp(- (logit + np.random.normal(scale=epsilon, size=len(ind)))))

    print(name, 'True AUC:', compute_auc(pclick, y), 'Noisy AUC:', compute_auc(noisy_pclick, y))

    # Build table with auctionid, dataid, and pclick
    df = pandas.DataFrame(zip(ind, pclick, noisy_pclick), columns=['SampleId', 'TruePClick', 'PClick'])
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
        layouts = np.random.randint(1, max_slate, len(ids))
        layout_df = pandas.DataFrame(zip(ids, layouts), columns=['AuctionId', 'Layout'])
        df = df.join(layout_df, on='AuctionId', how='inner', lsuffix='', rsuffix='_dup')
        df.drop(list(df.filter(regex='_dup$')), axis=1, inplace=True)
        df = df[df['Position'] < df['Layout']]
    else:
        df['Layout'] = df.groupby('AuctionId')['AuctionId'].transform('count')

    # Rank by PClick, then cascade to generate clicks
    df['Uniform'] = np.random.uniform(size=len(df))
    df['WouldClick'] = np.where(df['Uniform'] <= df['TruePClick'], 1, 0)
    df['Click'] = 0
    df.loc[df["WouldClick"].ne(0).groupby(df['AuctionId']).idxmax(),'Click']=1
    df['Click'] = df['Click'] * df['WouldClick']

    df.drop(columns=['Uniform', 'WouldClick', 'RankingPClick', 'TruePClick'], inplace=True)

    return df, seed+1

def construct_auction_dataset(dataset):
    X = np.hstack((dataset['auctions'][['PClick', 'Position', 'Layout']], dataset['X'][dataset['auctions']['SampleId']]))
    y = dataset['y'][dataset['auctions']['SampleId']]

    return X, y

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s', '--seed', default=16, type=int, help='Random seed for repro')
#     # Settings for true pclick distribution
#     parser.add_argument('--n_tr_samples', default=30000, type=int, help='Training Samples')
#     parser.add_argument('--n_te_cnt_samples', default=100000, type=int, help='Control Testing Samples')
#     parser.add_argument('--n_te_trt_samples', default=100000, type=int, help='Treatment Testing Samples')
#     parser.add_argument('--n_te_rnd_samples', default=100000, type=int, help='Treatment Randomized Samples')
#     parser.add_argument('--n_features', default=200, type=int, help='Features for distribution')
#     parser.add_argument('--n_informative', default=150, type=int, help='Relevant Features')
#     parser.add_argument('--n_clusters_per_class', default=10, type=int, help='Clusters per class')
#     parser.add_argument('--class_sep', default=3, type=float, help='Class separation')

#     parser.add_argument('--oracle_n_estimators', default=100, type=int, help='Number of estimators for Oracle')
#     parser.add_argument('--oracle_min_samples_leaf', default=100, type=int, help='Minimum number of samples for Oracle to use for labeling')

#     parser.add_argument('--auction_n_estimators', default=100, type=int, help='Number of estimators for Auction pclick models')
#     parser.add_argument('--auction_min_samples_leaf', default=10000, type=int, help='Minimum number of samples for Auction pclick models to use for labeling')

#     parser.add_argument('--auction_size', default=20, type=int, help='Size of Auction')
#     parser.add_argument('--n_rnd_auction', default=10000, type=int, help='Number of randomized auctions')
#     parser.add_argument('--n_auctions', default=100000, type=int, help='Number of auctions for control and treatment')
#     parser.add_argument('--epsilon', default = .5, type=float, help='Noise to add to pclicks prior to sorting')

#     parser.add_argument('--control_reserve', default=.6, type=float, help='Reserve on Control Flight')
#     parser.add_argument('--treatment_reserve', default=.7, type=float, help='Reserve on Treatment Flight')
#     parser.add_argument('--max_slate', default=5, type=float, help='Maximum slate size')
#     args = parser.parse_args()

#     print('Arguments', args)
#     X, y =  make_classification(  n_samples=args.n_tr_samples + args.n_te_cnt_samples + args.n_te_trt_samples + args.n_te_rnd_samples
#                                 , n_features=args.n_features
#                                 , n_informative=args.n_informative
#                                 , n_clusters_per_class=args.n_clusters_per_class
#                                 , class_sep=args.class_sep
#                                 , random_state=args.seed)
#     args.seed += 1

#     print('Splitting Datasets')
#     samples = np.array([0, args.n_tr_samples, args.n_te_cnt_samples, args.n_te_trt_samples, args.n_te_rnd_samples])
#     datasets = {name : {'name':name, 'start':start, 'end':end, 'samples':samples, 'auction_type':auction_type, 'reserve':reserve} for name, start, end, samples, auction_type, reserve in zip(['oracle', 'cnt', 'trt', 'rnd'], np.cumsum(samples), np.cumsum(samples)[1:], samples[1:], [None, 'greedy', 'greedy', 'random'], [None, args.control_reserve, args.treatment_reserve, 0])}

#     for dataset, info in datasets.items():
#         print(dataset, info['start'], info['end'], info['samples'])
#         info['X'], info['y'] = X[info['start']:info['end'], :].copy() , y[info['start']:info['end']].copy()

#     # Creates an oracle pclick that ignores position and simply observes c/nc
#     oracle, args.seed = train_rf(datasets['oracle']['X'], datasets['oracle']['y'], seed=args.seed, n_estimators=args.oracle_n_estimators, min_samples_leaf=args.oracle_min_samples_leaf)
#     print('Oracle Created -- Test AUC Control:', compute_model_auc(oracle, datasets['cnt']['X'], datasets['cnt']['y']))
#     print('Oracle Created -- Test AUC Treatment:', compute_model_auc(oracle, datasets['trt']['X'], datasets['trt']['y']))

#     # Run selection
#     datasets['rnd']['ind'], args.seed = run_selection(args.seed, datasets['rnd']['samples'], args.auction_size, args.n_rnd_auction)
#     datasets['cnt']['ind'], args.seed = run_selection(args.seed, datasets['cnt']['samples'], args.auction_size, args.n_auctions)
#     datasets['trt']['ind'], args.seed = run_selection(args.seed, datasets['trt']['samples'], args.auction_size, args.n_auctions)

#     # Run the auction
#     datasets['rnd']['auctions'], args.seed = run_auction(datasets['rnd'], args.seed, oracle, args.epsilon, args.auction_size, args.n_rnd_auction, args.max_slate)
#     datasets['cnt']['auctions'], args.seed = run_auction(datasets['cnt'], args.seed, oracle, args.epsilon, args.auction_size, args.n_auctions, args.max_slate)
#     datasets['trt']['auctions'], args.seed = run_auction(datasets['trt'], args.seed, oracle, args.epsilon, args.auction_size, args.n_auctions, args.max_slate)

#     print('Train RF Models')
#     rnd_model, args.seed = train_rf(*construct_auction_dataset(datasets['rnd']), seed=args.seed, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)
#     cnt_model, args.seed = train_rf(*construct_auction_dataset(datasets['cnt']), seed=args.seed, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)
#     trt_model, args.seed = train_rf(*construct_auction_dataset(datasets['trt']), seed=args.seed, n_estimators=args.auction_n_estimators, min_samples_leaf=args.auction_min_samples_leaf)

#     print('Eval rnd on cnt', compute_model_auc(rnd_model, *construct_auction_dataset(datasets['cnt'])))
#     print('Eval rnd on trt', compute_model_auc(rnd_model, *construct_auction_dataset(datasets['trt'])))
#     print('Eval cnt on cnt', compute_model_auc(cnt_model, *construct_auction_dataset(datasets['cnt'])))
#     print('Eval cnt on trt', compute_model_auc(cnt_model, *construct_auction_dataset(datasets['trt'])))
#     print('Eval trt on cnt', compute_model_auc(trt_model, *construct_auction_dataset(datasets['cnt'])))
#     print('Eval trt on trt', compute_model_auc(trt_model, *construct_auction_dataset(datasets['trt'])))
