from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import timeit
import warnings
import numpy as np
from catboost import CatBoostClassifier, Pool
from parametersOfModels import PARAMDISTRIBUTIONS, DEFAULTPARAMS_dict, all_metrics_classifications

warnings.filterwarnings('ignore')

def RandomSearchCVCatboost(model, X, y, 
                          cv : int = 3,
                          param_grid: dict = {},
                          random_samples: int = 5,
                          scoring_cv = accuracy_score,
                          high_is_better = True, # Для accuracy мы хотим, чтобы было больше, а для mse меньше 
                          verbose = 1,
                          balanced = False,
                          ):
  """
  returned the best_model fit and best_params
  Parameters:
    Model-similar to catboost (Regressor, Classifier)
    X, y - np.as array or pd. DataFrame
    param_grid-parameter grid
    scoring_cv-function that measures the quality of the classification or regression model
    high_is_better - what to look for the minimum or maximum of the scoring function
    
    other optional parameters
  Returns:
    Returns the best model and best parameters
  A method that allows you to search for hyperparameters for the catboost type of models.
  """

  if(isinstance(X, pd.DataFrame)):
    X = X.values
    y = y.values

  scores_history_tr = {}
  scores_history_ts = {}
  history_params = {}

  for sample in range(random_samples):
    # -- generate point in space parameter -- #
    # -- uniform distribution-- #
    params_sample = {key : np.random.choice(values, size= 1)[0] 
                            for key, values in param_grid.items()}

    history_params[sample] = params_sample                                     
    m = model(**params_sample)

    if(balanced):
      weights = compute_class_weight(class_weight = 'balanced',
                               classes = np.unique(y),
                               y = y)
      m.set_params(class_weights =  weights) 
    kf = StratifiedKFold(n_splits = cv, shuffle = True)
    train_score = []
    test_score = []
    for idx_tr, idx_ts in kf.split(X, y):
      X_tr, X_ts =  X[idx_tr], X[idx_ts]
      y_tr, y_ts = y[idx_tr], y[idx_ts]
      # -- create test pool and train pool -- #
      train_pool = Pool(
          X_tr, y_tr
      )
      test_pool = Pool(
          X_ts, y_ts
      )
      m.fit(train_pool, verbose = 0)
      y_pred_tr = m.predict(train_pool)
      y_pred_ts = m.predict(test_pool)
      train_score.append(scoring_cv(y_tr, y_pred_tr))
      test_score.append(scoring_cv(y_ts, y_pred_ts))
    
    train_score_np = np.asarray(train_score)
    test_score_np = np.asarray(test_score)

    train_info = {'mean_cv' : train_score_np.mean(), 
     'std_cv': train_score_np.std()}

    test_info = {'mean_cv' : test_score_np.mean(), 
     'std_cv': test_score_np.std()}

    if(verbose > 0):
      print('sample: {}, params: {}\ntrain_score: {}\ntest_score: {}'.format(
         sample + 1, params_sample, train_info, test_info, 
      ))


    scores_history_tr[sample] = train_info
    scores_history_ts[sample] = test_info

  # -- find the best score -- #
  # -- we want to minimize score -- # 
  mean_ts = np.asarray([value['mean_cv'] 
                        for key, value in scores_history_ts.items()])
    
  if(high_is_better):
    best_sample = mean_ts.argmax()
  else:
    best_sample = mean_ts.argmin()

  best_score = mean_ts[best_sample]

  if(verbose > 0):
    print('\nFinal:\nbest_validate_score: {}\nbest_params: {}'.format(
        best_score, history_params[best_sample],
    ))

  best_model = model(**history_params[best_sample])
  train_pool = Pool(X, y)
  best_model.fit(train_pool,
                 verbose = 0)
  return best_model, history_params[best_sample]
  

class AutoMLBinaryClassification2edition(object):

  """
  The class implements Auto ML. With common methods such as init-defines training type parameters and fit, predict.
  """
  
  def __init__(self, 
               defaultparams: bool = True, 
               allalgo : bool = False,
               balanced : bool = True,
               verbose = 0,
               blending = False,
               scoring_function = accuracy_score):
    
    """
    Parameters: 
    Returns:
    
    """
    
    self._defaultparams = defaultparams
    self._allalgo = allalgo

    self._one_model = None
    self._several_models = []

    self._balanced = balanced
    self._verbose = verbose
    self._blending = blending

    self._scoring_function = scoring_function

    self._names_models = ['logreg', 
                         'RF', 'SGDclf',
                         'ExtraTrees']

  def fit(self, X, y):
    """
    The fit method optionally allows you to count one model with default parameters (with Randomsearch search),
    several models (randomForest, LogRegression, SGDClassifier, ExtraTree). 
    And it is also possible to search for optimal hyperparameters for each model by Randomsearch.
    The situation when classes are unbalanced is also taken into account.
    """
    if(self._balanced):
        weights = compute_class_weight(class_weight = 'balanced',
                               classes = np.unique(y),
                               y = y)
    # -- one model with default params -- #
    if(self._defaultparams and not self._allalgo):
      start = timeit.default_timer()
      train_data = Pool(data = X, 
                        label = y)

      model = CatBoostClassifier(eval_metric='AUC',
                                 class_weights = weights if self._balanced else None,
                                 verbose = self._verbose)
      model.fit(train_data)
      if(self._verbose):
        print('Fited model CatboostClassifier with default parameters (time): {:.4f} sec'.format(timeit.default_timer() - start))
      scores = model.eval_metrics(train_data, metrics = 'AUC')
      self._one_model = model
    # -- tune (randomSearchCV) catboost classifiers -- #
    elif(not self._allalgo and not self._defaultparams):
      start = timeit.default_timer()
      parameter_space_cat_boost = {
              'iterations':  np.arange(50, 250, 25),
              'depth':  np.arange(2, 7, 2),
              'learning_rate': np.linspace(0.01, 0.5, 5),
              'min_data_in_leaf' :  np.linspace(1, 50, 5),
              'l2_leaf_reg' :  np.linspace(0, 5, 1),
      }
      best_catboost, best_params = RandomSearchCVCatboost(CatBoostClassifier, 
                                                          X, y, 
                                                          cv = 3,
                                                          param_grid = parameter_space_cat_boost,
                                                          random_samples = 5,
                                                          scoring_cv = self._scoring_function
                                                          )
      
      self._one_model = best_catboost
      if(self._verbose):
        print('best params RandomGridSearch: ', best_params)
        print('Fited model CatboostClassifier with tuned parameters (RandomSearch) (time): {:.4f} sec'.format(timeit.default_timer() - start))
      # -- several models and with default params -- #
    elif(self._allalgo and self._defaultparams):
      start_common = timeit.default_timer()
      models = [LogisticRegression(), RandomForestClassifier(), 
                SGDClassifier(), ExtraTreesClassifier()]
      linear_models = [True, False, True, True, False]
      for idx, (flaglin, model) in enumerate(zip(linear_models, models)):
        print("name: {}".format(self._names_models[idx]))
        model = model.set_params(**DEFAULTPARAMS_dict[self._names_models[idx]])
        start_current = timeit.default_timer()
        if(flaglin):
          # -- only for linear models -- #
          self._scaler = StandardScaler()
          X_scaler = self._scaler.fit_transform(X)
        else:
          X_scaler = X
        if(self._balanced and model.get_params().get('class_weight', -1) != -1):
          model.set_params(class_weight = 'balanced')
        model.fit(X, y)
        if(self._verbose):
          print('fited model: {} (time) : {:.3f} sec'.format(self._names_models[idx], 
                                                         timeit.default_timer() - start_current))
          print('-'*30)
        # -- write model -- #
        self._several_models.append(model)
      if(self._verbose):
        print('fitted all models with default parameters (time) : {:.3f} sec'.format(timeit.default_timer() - start_common))
    # -- find hyperparameters all models -- #
    elif(self._allalgo and not self._defaultparams):
      # -- I use simple RandomSearchGrid -- #
      start_common = timeit.default_timer()
      models = [LogisticRegression(), RandomForestClassifier(), 
                SGDClassifier(), ExtraTreesClassifier()]
      linear_models = [True, False, True, True, False]
      for idx, (flaglin, model) in enumerate(zip(linear_models, models)):
        start_current = timeit.default_timer()
        kf = StratifiedKFold(3, shuffle = True)
        if(self._balanced and model.get_params().get('class_weight', -1) != -1):
          model.set_params(class_weight = 'balanced')
        
        rs = RandomizedSearchCV(model, param_distributions = PARAMDISTRIBUTIONS[self._names_models[idx]],
                           n_iter = 5, scoring = make_scorer(self._scoring_function),
                           n_jobs = -1, 
                           cv = kf)
        rs.fit(X, y)
        if(self._verbose):
          print('tuned model: {} RandomSearchCV (time) : {:.3f} sec'.format(self._names_models[idx], 
                                                         timeit.default_timer() - start_current))
          print('-'*30)
        self._several_models.append(rs.best_estimator_)
      if(self._verbose):
        print('tuned all models RandomSearchCV (time) : {:.3f} sec'.format(timeit.default_timer() - start_common))

  def predict(self, X, blending = True):
    
    """
    Parameters:
    X - np.asarray or pd.DataFrame 
    blending - optional parameters for majority vote 
    Predict and Predict Probe can be done for multiple models by majority vote and just output for each model individually. 
       In Predict_Proba for several models, for simplicity, I decided to add up the estimates of class accessories. 
       I wanted to make Stacking for many models, but there was not enough time, so I did not complicate it."""
    
    if(not self._allalgo):
      X_pool = Pool(X)
      return self._one_model.predict(X_pool)
    else:
      preds = {}
      assert(self._several_models is not None), "Need fit models before call predict!"
      y_labels = np.empty((self._several_models.__len__(), X.shape[0]), dtype = np.int)
      for idx, m in enumerate(self._several_models):
        y_labels[idx, :] = m.predict(X)
      if(self._blending):
        # -- majority vote -- #
        return np.apply_along_axis(lambda col: np.bincount(col).argmax(), 0, y_labels)
      else:
        for idx, names in enumerate(self._names_models):
          preds[names] = y_labels[idx, :]
        return preds

  def predict_proba(self, X):
    """
    In Predict_Proba for several models, for simplicity, I decided to add up the estimates of class accessories.
    """
    if(not self._allalgo):
      X_pool = Pool(X)
      return self._one_model.predict(X_pool)
    else:
      # -- average -- #
      preds = np.empty((X.shape[0], 2))
      for idx, m in enumerate(self._several_models):
        try:
          preds += m.predict_proba(X)
        except Exception as e:
          pass
      
      return preds/len(self._several_models)
