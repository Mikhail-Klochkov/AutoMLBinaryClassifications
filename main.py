import pandas as pd
import os 
import warnings
from utils import load_data
import argparse
warnings.filterwarnings('ignore')
from preprocessing import preprocessing
from validation import crossValidationKfold
from autoMLClass import RandomSearchCVCatboost, AutoMLBinaryClassification2edition

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -- loader is folder where exist file csv -- #
    parser.add_argument('--loader',
                        required = True,
                        help = 'It is loader which contain train and test data'
                        )
    # -- name of train file csv -- #
    parser.add_argument('--train',
                        required = True,
                        help = 'Name of train data .csv'
                        )
    # -- mode (binary, multiclass)  (You can later implement) -- #
    parser.add_argument('--mode', 
                        choices = ['binary_classification', 'multi_classificaton'],
                        required = False,
                        help = 'it is defined type of machine learning tasks')

    names_csv_files = []
    names_another_files = []
    current_directory = os.path.dirname(os.path.realpath(__file__))
    args = parser.parse_args()
    # -- Searches the loader folder for all csv files -- #
    print(f"current path: {current_directory}")
    for root, dirs, files in os.walk(top = current_directory + "/" + args.loader):
        print(root, dirs)
        for file in files:
            if(file.find("csv") != -1):
                names_csv_files.append(root + "/" + file)
            else:
                names_another_files.append(root + "/" + file)

    
    if(args.train.find('csv') != -1):
        # -- load data from loader -- #
        df = load_data(names_csv_files[0])
        # -- shuffle -- #
        df = df.sample(frac = 1).reset_index(drop=True)
        # -- split on train and test -- #
        split_index = int(0.7 * df.shape[0])
        df_test = df.iloc[split_index: ]
        df_train = df.iloc[:split_index] 
        # -- prepare data -- #
        df_train, df_test = preprocessing(df_train, 
                       df_test, 
                       inplace = False,
                       feature_importance = False,
                       verbose = True)

        print('Preprocessed and Feature engineering is Done!')
        print('-'*30)
        # -- fit some models -- #
        X, y = df_train.drop(columns = ['target']), df_train['target']
        if(df_train.shape[0] > 5000 and df_train.shape[0] > 5000):
            X, y = X[:5000], y[:5000]
        
        params_ = {'balanced' : True,
                   'defaultparams' : True,
                   'verbose' : False,
                   'allalgo': True,
                   'blending' : True,
            }
        # -- fit and validations with default params, with several models -- #
        # -- with blending all models and Take into account class balancing (I didn't have time to make it another parameter)  -- #
        means_metrics = crossValidationKfold(AutoMLBinaryClassification2edition, 
                     X.iloc[:5000, :], y[:5000],
                     params_automl = params_,
                     allmetrics = True,
                     verbose = True,
                     )
        print('The model is trained and the quality is calculated on the stratified cross-validation')
        print('-'*30)
        print(means_metrics)
    print(f"shape: {df_train.shape}")
