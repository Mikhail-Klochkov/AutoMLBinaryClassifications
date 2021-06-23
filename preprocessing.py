import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

MIN_ONE_HOT_UNIQUE = 20
BIG_DATASET_SIZE = 1024 * 1024 * 700

def preprocessing(df_train: pd.DataFrame, 
                  df_test: pd.DataFrame,
                  colnametarget: str = "target", 
                  missing_strategy: str = 'mode_median', 
                  feature_importance = False,
                  inplace: bool = True, 
                  verbose: bool = True): 
  """
  Parameters:
    df_train - Train data
    df_test - Test data 
    colnametarget - Name target column
    missing strategy - Strategy for filling in missing values (Implemented only one simple version of the mod and median)
    feature_importance - If the user wants to output the values of the importance of the features
    inplace - Whether to create copies
    verbose - Output of intermediate information

    Returns:
    If the inplace parameter = True then None otherwise preprocessed data

    -----------------------------------------------------

  """
  tr_size = df_train.shape[0]
  drop_columns = []
  if(not inplace):
    df_train = df_train.copy()
    df_test = df_test.copy()

  catcols = df_train.dtypes[df_train.dtypes == 'object'].index.tolist()
  # -- detect cat features -- #
  potentially_categorical = [col
                            for col in df_train.columns 
                            if (len(df_train[col].unique()) < tr_size / 100) \
                            and (col not in catcols) and \
                            tr_size / 10 > 100 
                            and col != colnametarget]
  catcolsall = catcols + potentially_categorical    
  print('Number of catogorical features: ', len(catcolsall))                        
  # -- drop constant columns -- #
  constant_columns_or_ID = [
                      col_name 
                      for col_name in df_train.columns
                      if (df_train[col_name].nunique() == 1) or 
                         (df_train[col_name].nunique() == tr_size)
                      ] 
  df_train.drop(columns = constant_columns_or_ID, 
                axis = 1, 
                inplace = True)
  df_test.drop(columns = constant_columns_or_ID, 
                axis = 1, 
                inplace = True)

  drop_columns += constant_columns_or_ID
  # -- transform datetime objects -- #
  datetimecols = df_train.select_dtypes('datetime').columns.tolist()

  for col in datetimecols:
    df_train["num_weekday={}".format(col)] = df_train[col].apply(lambda x: x.weekday())
    df_train['num_month_{}'.format(col)] = df_train[col].apply(lambda x: x.month)
    df_train['num_day_{}'.format(col)] = df_train[col].apply(lambda x: x.day)
    df_train['num_year_{}'.format(col)] = df_train[col].apply(lambda x: x.year)

    df_test["num_weekday={}".format(col)] = df_test[col].apply(lambda x: x.weekday())
    df_test['num_month_{}'.format(col)] = df_test[col].apply(lambda x: x.month)
    df_test['num_day_{}'.format(col)] = df_test[col].apply(lambda x: x.day)
    df_test['num_year_{}'.format(col)] = df_test[col].apply(lambda x: x.year)

  # -- drop datetime cols -- #
  df_train.drop(columns = datetimecols, inplace = True)
  df_test.drop(columns = datetimecols, inplace = True)
  catcolsall = list(set(catcolsall).difference(datetimecols))

  drop_columns += datetimecols

  # -- missing values -- #
  # -- without target -- #
  for col in df_train.drop(columns = [colnametarget]).columns:
    # -- smaller than 15 % -- #
    if(df_train[col].isna().sum()/tr_size <= 0.15):
      if(col in catcols):
        if missing_strategy == 'mode_median':
          fillvalues = df_train[col].mode(dropna = True)[0]
      else:
        if missing_strategy == 'mode_median':
          fillvalues = df_train[col].median()
      df_train[col].fillna(value = fillvalues, inplace = True)
      df_test[col].fillna(value = fillvalues, inplace = True)
    else:
      df_train.drop(columns = [col], inplace = True)
      df_test.drop(columns = [col], inplace = True)
      drop_columns += col
  
  # -- numeric variables (for linear model) or maybe outliers only for -- #
  koef = 1.5
  allnumberoutliers = 0

  for col in df_train.drop(columns = [colnametarget] + catcolsall).columns:
    q1 = df_train[col].quantile(0.25)
    q2 = df_train[col].quantile(0.75)
    irq = q2 - q1
    outliers_idxs = df_train[col][(df_train[col] < q1 - irq * koef) | (df_train[col] > q2 + irq * koef)].index
    # -- only for 3 percents -- #
    if((len(outliers_idxs) / tr_size) * 100 < 3):
      df_train.drop(index = outliers_idxs, inplace = True)
      allnumberoutliers += len(outliers_idxs)
      if(verbose):
        print(f'column: {col} percent of outliers: {(len(outliers_idxs) / tr_size) * 100} %')
        print('-'*20)
  if(verbose):
    print(f'delete {(allnumberoutliers / tr_size) * 100} % outliers')
    print("-"*30)

  onehotvariables = []
  num_cols_before = df_train.shape[1]
  # -- encode category features -- #
  is_big = df_train.memory_usage().sum() > BIG_DATASET_SIZE
  if(is_big):
    # -- for very big datasets we need delete some columns -- #
    # -- delete low correlated feature with target -- #
    min_features_count = min(df_train[1], 
                             int(df_train.shape[1] / (df_train.memory_usage().sum() / BIG_DATASET_SIZE)))
    correlations = {col : np.corrcoef(df_train[col], df_train[colnametarget])[0, 1] 
                    for col in df_train.select_dtypes("number").columns 
                    if col != colnametarget}
    sortcolsbypearsoncorrelations = [k for k, _ in sorted(correlations.items(), key = lambda x: x[1])]
    print('file so big, we should delete some columns with small pearson correlation')
    print("before: {} after: {} ".format(df_train.shape[1], min_features_count))
    df_train.drop(columns = sortcolsbypearsoncorrelations[:-min_features_count],
                  inplace = True) 
    drop_columns += sortcolsbypearsoncorrelations[:-min_features_count]

  assert(df_train.isna().sum(axis = 0).sum() == 0), "Should be no missing values in train data"
  assert(df_test.isna().sum(axis = 0).sum() == 0), "Should be no missing values in test data"

  # -- feature selection -- #
  if(feature_importance):
    df_train_copy = df_train.copy()
    for col in catcolsall:
      le = LabelEncoder()
      df_train_copy[col] = le.fit_transform(df_train_copy[col])
      
    X, y = df_train_copy.drop(columns = [colnametarget]), df_train_copy[colnametarget]
    y_label = LabelEncoder().fit_transform(y)
    assert(X.select_dtypes("number").columns.__len__() == X.shape[1]), "Before train RF we should have all numeric data!"
    rf = RandomForestClassifier(n_estimators = 25, 
                                 max_features = int(np.sqrt(X.shape[1])/2),
                                criterion = "entropy",
                                min_samples_leaf = 5)
    rf.fit(X, y_label)
    results = permutation_importance(rf,
                                     X, 
                                     y_label, 
                                     n_repeats = 3)
    if(verbose):
        print("feature importance: ", results.importances_mean)
        print('-'*30)
    
    # -- for economy memory -- #
    del df_train_copy, X, y

  for col in catcolsall:
    col_unique = df_train[col].unique()
    # -- onehot -- #
    if len(col_unique) <= MIN_ONE_HOT_UNIQUE and len(col_unique) > 2:
      cat_cols_num_unique = {}
      for value in col_unique:
        df_train[f"onehot_{col}={str(value)}"] = (df_train[col] == value).astype(int)
        df_test[f"onehot_{col}={str(value)}"] = (df_test[col] == value).astype(int)
      
      df_train.drop(columns = [col], inplace = True)
      df_test.drop(columns = [col], inplace = True)
    # -- LabelEncoder -- #
    else:
      le = LabelEncoder()
      df_train[col] = le.fit_transform(df_train[col])
      # -- this can be previosly unseen data -- #
      try:
        df_test[col] = le.transform(df_test[col])
      except (KeyError, ValueError):
        mode = df_train[col].mode()
        mapuniqe2label = {unique: label 
                          for unique, label in zip(col_unique, 
                                             np.arange(len(col_unique)))}
        # -- if we see unseen data take most common -- #                                            
        df_test[col] = df_test[col].apply(lambda x: mapuniqe2label.get(x, 
                                                                      mode))
  # -- encode target -- #
  le_label = LabelBinarizer()
  df_train[colnametarget] = le_label.fit_transform(df_train[colnametarget])
  df_test[colnametarget] = le_label.transform(df_test[colnametarget])
  
  if(not inplace):
    return df_train, df_test
  else:
    return None
