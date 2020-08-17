import argparse
import numpy as np
from simulation.util import *
from ctrf.metrics import *
from ctrf.models import *
from ctrf.auction import *
import os,pickle

###Used for Simulation Data
def update_results(model_name,results,test_X,test_Y):
    results[model_name]['auc'].append(compute_model_auc(eval(model_name),test_X,test_Y))
    results[model_name]['bias'].append(compute_model_bias(eval(model_name),test_X,test_Y))    
    results[model_name]['rig'].append(compute_model_rig(eval(model_name),test_X,test_Y))
    results[model_name]['f1_score'].append(compute_model_f1(eval(model_name),test_X,test_Y))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=16, type=int, help='Random seed for repro')
    # Settings for true pclick distribution
    parser.add_argument('--n_rnd', default=1000, type=int, help='Random Data Size')
    parser.add_argument('--n_log', default=3500, type=int, help='Log Data Size')
    parser.add_argument('--n_test', default=2000, type=int, help='Random Data Size')
    parser.add_argument('--p', default=40, type=int, help='Feature Size')
    parser.add_argument('--scenario_log', default=2, type=int, help='Log data Scenario')
    parser.add_argument('--scenario_test', default=2, type=int, help='Testing data Scenario')
    parser.add_argument('--r_log', default=0.7, type=float, help='Sampling Bias for Log data')
    parser.add_argument('--r_test', default=0.7, type=float, help='Sampling Bias for Testing data')
    parser.add_argument('--n_experiments', default=200, type=int, help='Number of Experiments')
    args = parser.parse_args()
    print('Arguments', args)
    
    model_list=['lr_model','lr_weight_model','gbdt_model','gbdt_weight_model','rndrf_model','cntrf_model','trtrf_model','combinerf_model','ctrf_model']
    results={name:{metric:[] for metric in ['auc','f1_score','bias','rig']} for name in model_list}
 
    if os.path.isdir('simu_results'): 
        print('')
    else:
        os.mkdir("simu_results")
        print("directory added")

    path="simu_results/"
    os.chdir(path)
    result_name='_'.join(["results",str(args.n_log),str(args.p),str(args.scenario_test),str(int(100*args.r_test))])

    for i in range(args.n_experiments):
        ##Data Generating
        print ('------------------------------')
        print ('Run %d th experiments:' % (i+1))
        random_data=simu_confounding_data(p=args.p,n=args.n_rnd,scenario=1,r=0.5)
        log_data=simu_confounding_data(p=args.p,n=args.n_log,scenario=args.scenario_log,r=args.r_log)
        testing_data=simu_confounding_data(p=args.p,n=args.n_test,scenario=args.scenario_test,r=args.r_test)
        #IPW Calculation
        ipw_weights=calculate_weight(log_data['X'],testing_data['X'])

        ##Logistic Regression
        lr_model,args.seed=train_lr_model(log_data['X'],log_data['Y'],seed=args.seed)
        ##Logistic Regression with IPW adjustment
        lr_weight_model,args.seed=train_lr_weight_model(log_data['X'],log_data['Y'],ipw_weights,seed=args.seed)
        ##GBDT
        gbdt_model,args.seed=train_gbdt_model(log_data['X'],log_data['Y'],seed=args.seed)
        ##GBDT with IPW adjustment
        gbdt_weight_model,args.seed=train_gbdt_weight_model(log_data['X'],log_data['Y'],ipw_weights,seed=args.seed)
        #RF on randomized
        rndrf_model,args.seed=train_rf(random_data['X'],random_data['Y'],seed=args.seed)
        #RF on Log
        cntrf_model,args.seed=train_rf(log_data['X'],log_data['Y'],seed=args.seed)
        #Oracle RF
        trtrf_model,args.seed=train_rf(testing_data['X'],testing_data['Y'],seed=args.seed)
        #Combined RF
        combinerf_model,args.seed=train_combine_rf(random_data['X'],random_data['Y'],log_data['X'],log_data['Y'],seed=args.seed)       
        #CTRF
        ctrf_model,args.seed=train_ctrf(random_data['X'],random_data['Y'],log_data['X'],log_data['Y'],rndrf_model,combine=1,seed=args.seed)

        #Record Results
        for model_name in model_list:
            update_results(model_name,results,testing_data['X'],testing_data['Y'])

        print ('Finish %d th experiments:' % (i+1))
    
    f = open(result_name+".pkl","wb")
    pickle.dump(results,f)
    f.close()