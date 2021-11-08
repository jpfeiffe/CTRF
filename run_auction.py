import argparse
import copy
import itertools
import numpy as np
import time

from multiprocessing import Pool

from sklearn.datasets import make_classification
from ctrf.auction import run_auction_evaluation


def submit_job(param):
    args = param[0]
    seed = param[1]
    # trial = param[2]
    welfare_delta = param[3]

    args.welfare_delta = welfare_delta
    args.treatment_welfare_inc = args.control_welfare_inc * args.welfare_delta
    args.seed = seed
    run_auction_evaluation(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=50, type=int, help='Random seed for repro')
    # Settings for true pclick distribution
    parser.add_argument('--n_tr_samples', default=30000, type=int, help='Training Samples')
    parser.add_argument('--n_te_cnt_samples', default=100000, type=int, help='Control Testing Samples')
    parser.add_argument('--n_te_trt_samples', default=100000, type=int, help='Treatment Testing Samples')
    parser.add_argument('--n_te_rnd_samples', default=100000, type=int, help='Treatment Randomized Samples')
    parser.add_argument('--n_features', default=6, type=int, help='Features for distribution')
    parser.add_argument('--n_informative', default=5, type=int, help='Relevant Features')
    parser.add_argument('--n_clusters_per_class', default=8, type=int, help='Clusters per class')
    parser.add_argument('--class_sep', default=.5, type=float, help='Class separation')

    parser.add_argument('--oracle_n_estimators', default=100, type=int, help='Number of estimators for Oracle')
    parser.add_argument('--oracle_min_samples_leaf', default=10, type=int, help='Minimum number of samples for Oracle to use for labeling')

    parser.add_argument('--auction_n_estimators', default=100, type=int, help='Number of estimators for Auction pclick models')
    parser.add_argument('--auction_min_samples_leaf', default=10, type=int, help='Maximum tree leaf nodes for Auction pclick models to use for labeling')

    parser.add_argument('--control_auction_size', default=20, type=int, help='Size of Auction')
    parser.add_argument('--treatment_auction_size', default=20, type=int, help='Size of Auction')
    # parser.add_argument('--n_rnd_auction', default=10000, type=int, help='Number of randomized auctions')
    # parser.add_argument('--n_auctions', default=100000, type=int, help='Number of auctions for control and treatment')

    parser.add_argument('--n_rnd_auction', default=10000, type=int, help='Number of randomized auctions')
    parser.add_argument('--n_auctions', default=100000, type=int, help='Number of auctions for control and treatment')

    parser.add_argument('--epsilon', default = 1.5, type=float, help='Noise to add to pclicks prior to sorting')

    parser.add_argument('--control_reserve', default=0, type=float, help='Reserve on Control Flight')
    parser.add_argument('--treatment_reserve', default=0, type=float, help='Reserve on Treatment Flight')

    parser.add_argument('--random_max_slate', default=3, type=float, help='Maximum slate size')
    parser.add_argument('--control_max_slate', default=3, type=float, help='Maximum slate size')
    parser.add_argument('--treatment_max_slate', default=3, type=float, help='Maximum slate size')

    parser.add_argument('--control_welfare', default=.8, type=float, help='Reserve on Control Flight')
    parser.add_argument('--treatment_welfare', default=.8, type=float, help='Reserve on Treatment Flight')
    parser.add_argument('--control_welfare_inc', default=1.11, type=float, help='Increase in welfare requirement per layout')
    parser.add_argument('--treatment_welfare_inc', default=None, type=float, help='Increase in welfare requirement per layout')

    parser.add_argument('--welfare_inc_delta', default=[.95, 1.05, .01], nargs=3, help='Test scale control_welfare_inc to treatment_welfare_inc')
    parser.add_argument('--welfare_delta', default=None)
    parser.add_argument('--trials', default=10)
    parser.add_argument('--timetag', default=None)

    parser.add_argument('--processes', default=8, type=int)
    args = parser.parse_args()

    args.timetag = int(time.time())


    welfare_incs = np.arange(*args.welfare_inc_delta)
    trials = list(range(args.trials))
    trial_X_welfare_inc = list(itertools.product(trials, welfare_incs))
    seeds = [args.seed*i for i in range(len(trial_X_welfare_inc))]

    params = [[copy.deepcopy(args), s, t_w[0], t_w[1]]  for s, t_w in zip(seeds, trial_X_welfare_inc)]

    pool = Pool(args.processes)
    
    pool.map(submit_job, params)
