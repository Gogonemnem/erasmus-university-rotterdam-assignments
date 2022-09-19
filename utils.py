import os
data_folder_path = "./data"

import pandas as pd

def load_data():
    df_train = pd.read_csv(f'{data_folder_path}/train.csv', delimiter=';', decimal=",")
    x_train = df_train.iloc[:,1:]
    y_train = df_train['target']

    x_test = pd.read_csv(f'{data_folder_path}/test.csv', delimiter=';', decimal=",")
    return x_train, y_train, x_test

from sklearn.model_selection import StratifiedKFold

def kfold_cv(X, y, k, H, cv_fun, random_state):
    """
    Do stratified k-fold cross-validation with a dataset, to check how a model behaves as a function
    of the values in H (eg. a hyperparameter such as tree depth, or polynomial degree).

    :param X: feature matrix.
    :param y: response column.
    :param k: number of folds.
    :param H: values of the hyperparameter to cross-validate.
    :param cv_fun: function of the form (X_train, y_train, X_valid, y_valid, h) to evaluate the model in one split,
        as a function of h. It must return a dictionary with metric score values.
    :param random_state: controls the pseudo random number generation for splitting the data.
    :return: a Pandas dataframe with metric scores along values in H.
    """
    kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = random_state)
    pr = []  # to store global results

    # for each value h in H, do CV
    for i, h in enumerate(H):
        scores = []  # to store the k results for this h
        # for each fold 1..K
        for train_index, valid_index in kf.split(X, y):
            # partition the data in training and validation
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # call cv_fun to train the model and compute performance
            fold_scores = cv_fun(X_train, y_train, X_valid, y_valid, h)
            scores.append(fold_scores)

        rowMeans = pd.DataFrame(scores).mean(axis = 0)  # average scores across folds
        pr.append(rowMeans)  # append to global results
        print(f'Training models: {i+1}/{len(H)}, with hyperparameters: {h}', end='\t\t\r')

    pr = pd.DataFrame(pr).assign(_h = H)
    return pr

