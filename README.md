# AutoMLBinaryClassifications
A simple library implementation for solving the binary classification problem. It contains useful functions that you can independently refine and make more flexible with a variety of managed parameters.


Each class and function contains a brief description of what it does and what parameters it accepts.

File preprocessing.py -contains a function that does all the data retraining actions (see in the file).

The auto Ml Class file contains the model class itself, as well as the Random Search C V helper function for CatboostClassifier/CatboostRegressor.

Parameters Of Models file-contains grids for searching for hyperparameters of models and default parameters.

The validation file contains a method for cross validation_scores-calculation of the classification quality metrics of interest.

In general, I wrote more in Colab, where it was more convenient to track the results, so in this repository there are only all methods and classes.

The library is still under development, it is the main framework. Then you can make the methods even deeper and more flexible.
