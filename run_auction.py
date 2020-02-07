import argparse
import numpy as np
from sklearn.datasets import make_classification
#
from ctrf.metrics import compute_auc, compute_model_auc
from ctrf.models import train_rf, train_ctrf
from ctrf.auction import run_selection, run_auction, construct_auction_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=16, type=int, help='Random seed for repro')
    # Settings for true pclick distribution
    parser.add_argument('--n_tr_samples', default=30000, type=int, help='Training Samples')
    parser.add_argument('--n_te_cnt_samples', default=100000, type=int, help='Control Testing Samples')
    parser.add_argument('--n_te_trt_samples', default=100000, type=int, help='Treatment Testing Samples')
    parser.add_argument('--n_te_rnd_samples', default=100000, type=int, help='Treatment Randomized Samples')
    parser.add_argument('--n_features', default=11, type=int, help='Features for distribution')
    parser.add_argument('--n_informative', default=10, type=int, help='Relevant Features')
    parser.add_argument('--n_clusters_per_class', default=10, type=int, help='Clusters per class')
    parser.add_argument('--class_sep', default=1, type=float, help='Class separation')

    parser.add_argument('--oracle_n_estimators', default=100, type=int, help='Number of estimators for Oracle')
    parser.add_argument('--oracle_min_samples_leaf', default=100, type=int, help='Minimum number of samples for Oracle to use for labeling')

    parser.add_argument('--auction_n_estimators', default=50, type=int, help='Number of estimators for Auction pclick models')
    parser.add_argument('--auction_max_leaf_nodes', default=100, type=int, help='Maximum tree leaf nodes for Auction pclick models to use for labeling')

    parser.add_argument('--auction_size', default=20, type=int, help='Size of Auction')
    parser.add_argument('--n_rnd_auction', default=10000, type=int, help='Number of randomized auctions')
    parser.add_argument('--n_auctions', default=100000, type=int, help='Number of auctions for control and treatment')
    parser.add_argument('--epsilon', default = .1, type=float, help='Noise to add to pclicks prior to sorting')

    parser.add_argument('--control_reserve', default=.5, type=float, help='Reserve on Control Flight')
    parser.add_argument('--treatment_reserve', default=.7, type=float, help='Reserve on Treatment Flight')
    parser.add_argument('--max_slate', default=5, type=float, help='Maximum slate size')
    args = parser.parse_args()

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
    datasets = {name : {'name':name, 'start':start, 'end':end, 'samples':samples, 'auction_type':auction_type, 'reserve':reserve} for name, start, end, samples, auction_type, reserve in zip(['oracle', 'cnt', 'trt', 'rnd'], np.cumsum(samples), np.cumsum(samples)[1:], samples[1:], [None, 'greedy', 'greedy', 'random'], [None, args.control_reserve, args.treatment_reserve, 0])}

    for dataset, info in datasets.items():
        print(dataset, info['start'], info['end'], info['samples'])
        info['X'], info['y'] = X[info['start']:info['end'], :].copy() , y[info['start']:info['end']].copy()

    # Creates an oracle pclick that ignores position and simply observes c/nc
    oracle, args.seed = train_rf(datasets['oracle']['X'], datasets['oracle']['y'], seed=args.seed, n_estimators=args.oracle_n_estimators, min_samples_leaf=args.oracle_min_samples_leaf)
    print('Oracle Created -- Test AUC Control:', compute_model_auc(oracle, datasets['cnt']['X'], datasets['cnt']['y']))
    print('Oracle Created -- Test AUC Treatment:', compute_model_auc(oracle, datasets['trt']['X'], datasets['trt']['y']))

    # Run selection
    datasets['rnd']['ind'], args.seed = run_selection(args.seed, datasets['rnd']['samples'], args.auction_size, args.n_rnd_auction)
    datasets['cnt']['ind'], args.seed = run_selection(args.seed, datasets['cnt']['samples'], args.auction_size, args.n_auctions)
    datasets['trt']['ind'], args.seed = run_selection(args.seed, datasets['trt']['samples'], args.auction_size, args.n_auctions)

    # Run the auction
    datasets['rnd']['auctions'], args.seed = run_auction(datasets['rnd'], args.seed, oracle, args.epsilon, args.auction_size, args.n_rnd_auction, args.max_slate)
    datasets['cnt']['auctions'], args.seed = run_auction(datasets['cnt'], args.seed, oracle, args.epsilon, args.auction_size, args.n_auctions, args.max_slate)
    datasets['trt']['auctions'], args.seed = run_auction(datasets['trt'], args.seed, oracle, args.epsilon, args.auction_size, args.n_auctions, args.max_slate)

    print('Train RF Models')
    rnd_rf, args.seed = train_rf(*construct_auction_dataset(datasets['rnd']), seed=args.seed, n_estimators=args.auction_n_estimators, max_leaf_nodes=args.auction_max_leaf_nodes)
    cnt_rf, args.seed = train_rf(*construct_auction_dataset(datasets['cnt']), seed=args.seed, n_estimators=args.auction_n_estimators, max_leaf_nodes=args.auction_max_leaf_nodes)
    trt_rf, args.seed = train_rf(*construct_auction_dataset(datasets['trt']), seed=args.seed, n_estimators=args.auction_n_estimators, max_leaf_nodes=args.auction_max_leaf_nodes)
    cnt_ctrf, args.seed = train_ctrf(*construct_auction_dataset(datasets['rnd']), *construct_auction_dataset(datasets['cnt']), seed=args.seed, n_estimators=args.auction_n_estimators, max_leaf_nodes=args.auction_max_leaf_nodes)
    trt_ctrf, args.seed = train_ctrf(*construct_auction_dataset(datasets['rnd']), *construct_auction_dataset(datasets['trt']), seed=args.seed, n_estimators=args.auction_n_estimators, max_leaf_nodes=args.auction_max_leaf_nodes)

    print('Eval rnd_rf on cnt', compute_model_auc(rnd_rf, *construct_auction_dataset(datasets['cnt'])))
    print('Eval rnd_rf on trt', compute_model_auc(rnd_rf, *construct_auction_dataset(datasets['trt'])))

    print('Eval cnt_rf on trt', compute_model_auc(cnt_rf, *construct_auction_dataset(datasets['trt'])))
    print('Eval trt_rf on cnt', compute_model_auc(trt_rf, *construct_auction_dataset(datasets['cnt'])))

    print('Eval cnt_ctrf on trt', compute_model_auc(cnt_ctrf, *construct_auction_dataset(datasets['trt'])))
    print('Eval trt_ctrf on cnt', compute_model_auc(trt_ctrf, *construct_auction_dataset(datasets['cnt'])))
