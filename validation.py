import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from parametersOfModels import all_metrics_classifications


def crossValidationKfold(automodel, 
                         X, y,
                         params_automl : dict = {},
                         score_function = accuracy_score,
                         cv : int = 3,
                         shuffle: bool = True,
                         verbose : bool = True,
                         allmetrics: bool = False):

  """
  The cross-classification method for our AutoMLBinaryClassification class.
  Parameters:
   1) model - our model implemented earlier
   2) X, y - np.ndarray prod.Data Frame with
   3) params_automl-model parameters for initializing the class object.
   4) score_function - quality function (metric) default = accuracy.
   5) cv-number of StratifiedKFold
   6) allmetrics-calculate all metrics (f1, recall, precision, roc_auc, accuracy and etc)
  Returns:
    if allmetrics is False - Returns two values mean the scoring_cv and std of the specifically passed metric
    else - Returns dict for all cv metrics (f1, precision, recall, precision, roc_auc).
  """
  if(isinstance(X, pd.DataFrame) or isinstance(y, pd.DataFrame)):
    X = X.values
    y = y.values
  skf = StratifiedKFold(n_splits = cv, 
                        shuffle = shuffle, 
                        random_state = 42)
  if(allmetrics):
    train_scores = {'accuracy' : [], 
                    'roc_auc': [], 
                    'f1' : [], 
                    'recall' : [], 
                    'precision': []}
    test_scores = {'accuracy' : [], 
                    'roc_auc': [], 
                    'f1' : [], 
                    'recall' : [], 
                    'precision': []}
  else:
    train_scores = np.empty((cv, ))
    test_scores = np.empty((cv, ))
  for idx, (idx_tr, idx_ts) in enumerate(skf.split(X, y)):
    X_tr, X_ts = X[idx_tr], X[idx_ts]
    y_tr, y_ts = y[idx_tr], y[idx_ts] 
    am = automodel(**params_automl)
    am.fit(X_tr, y_tr)
    if(not allmetrics):
      
      train_scores[idx] = score_function(am.predict(X_tr), y_tr)
      test_scores[idx] = score_function(am.predict(X_ts), y_ts)
      if(verbose):
        print('it: {} train score: {:.3f}, val score: {:.3f}'.format(idx, 
                                                                    train_scores[idx],
                                                                    test_scores[idx]))
    else:
      train_current = {}
      test_current = {}
      for name, metric in all_metrics_classifications.items():
        train_current[name] = metric(am.predict(X_tr), y_tr)
        test_current[name] = metric(am.predict(X_ts), y_ts)
        train_scores[name].append(train_current[name])
        test_scores[name].append(test_current[name])
        
      if(verbose):
        print('it: {} train scores: {}, val scores: {}'.format(idx, train_current,
                                                               test_current))

  if(not allmetrics):
    return test_scores.mean(), test_scores.std()
  else:
    # -- calculate means of all metrics-- #
    return dict(map(lambda kv: (kv[0], np.asarray(kv[1]).mean()), test_scores.items()))
