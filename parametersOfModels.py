
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

log_params = {
    'C' : 2,
}

sgd_params = {
    'alpha': 1,
    'penalty': 'l2',
}

DEFAULTPARAMS_dict = {'logreg': log_params,
                      'RF': rf_params,
                      'SGDclf': sgd_params,
                      'svm': svc_params,
                      'ExtraTrees': et_params}

# -- we define here gridCV for several simple algorithms for tuning -- #

param_grid_logisticregression = {
'C' : np.asarray([0.001,0.01,0.1,1,10,100,1000])
}

param_grid_randomforest = {
    'n_estimators' : np.arange(50, 301, 50),
    'max_depth' : np.arange(2, 7, 2),
    'min_samples_leaf': np.arange(2, 100, 20),
    'max_features' : ['auto', 'sqrt', 'log2'],
}

param_grid_extratree = {
        'n_estimators': np.arange(50, 301, 50),
        'min_samples_leaf': range(20,50,10),
        'min_samples_split': range(15,36,10),
    },

param_grid_sgdclassifier = {
    "loss" : ["hinge", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
}

param_grid_svc = {
               'C':[10, 100, 1000], 
              'gamma': [0.1, 1, 10], 
              'kernel': ['rbf', 'poly', 'linear']}

PARAMDISTRIBUTIONS = {'logreg': param_grid_logisticregression,
                      'RF': param_grid_randomforest,
                      'SGDclf': param_grid_sgdclassifier,
                      'svm': param_grid_svc,
                      'ExtraTrees': param_grid_extratree}


all_metrics_classifications = {'accuracy' : accuracy_score, 
                               'roc_auc': roc_auc_score, 
                               'f1' :f1_score, 
                               'recall' : recall_score, 
                               'precision': precision_score}