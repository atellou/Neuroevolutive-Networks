
import neuroevo as ne
import utility as ut
import numpy as np
import pandas as pd
import sys
import os
import sklearn.metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit

def one_run():

    act= 'vary'
    base = 'NEAT'
    config_name = 'config-recurrent'

    scores = {'columns': ['fitness', 'f1_score', 'recall', 'presicion', 'kappa']}
    act_functions = ['relu', 'sigmoid', 'tanh', 'vary']

    row, col = np.shape(X_train)  # No. Samples and Features
    n_targets = len(list(dict.fromkeys(y_train)))  # No. targets

    if act == 'vary':
        config = ut.modify_config(os.path.join(base, config_name),  # crate the config file
                                activation_def='relu', fitnes_th=1, n_inputs=col, n_outputs=n_targets,
                                activation_opt=act_functions[:-1])
    else:
        config = ut.modify_config(os.path.join(base, config_name),  # crate the config file
                                activation_def=act, fitnes_th=1, n_inputs=col, n_outputs=n_targets)
    fitness = ne.fitness_func(X_train,
                            y_train)  # create the object to store the train data and pass the fitness function
    winner, predictions, stats = ne.run(config,
                                    X_test, y_test, fitness,
                                    generations=10,
                                    save_check=0, save=0, save_best=0,
                                    verbose=1, view=1, feats=feature_names, targets=target_names)
    pred_targets = np.argmax(predictions, axis=1)
    recall = sklearn.metrics.recall_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
    f1_score = sklearn.metrics.f1_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
    presicion = sklearn.metrics.precision_score(y_test, pred_targets, labels=range(n_targets),
                                                average='weighted')
    kappa = sklearn.metrics.cohen_kappa_score(y_test, pred_targets, labels=range(n_targets))
    scores[act] = [winner.fitness, f1_score, recall, presicion, kappa]
    print('Termine ', os.path.join(base, act))
    print(pd.DataFrame(scores, index=scores['columns']).drop(['columns'], axis=1).T)


def new_evolution(): #Example case 2
    """This will run from zero all the process for more than one option of activation function. The directory that is used in this example 
    from the function files_tour was organized as following: 
                                                    -NEAT:
                                                        "Using 4 options (relu, sigmoid, tanh, varay)"
                                                        config_recurrent
                                                        -relu
                                                            -Check
                                                                relu_neat-checkpoint-1 
                                                                .
                                                                .
                                                                .
                                                                relu_neat-checkpoint-989
                                                            -Images
                                                                relu_fitnass.svg
                                                                relu_net_graph
                                                                relu_speciation.svg
                                                        -sigmoid
                                                            kind of above
                                                        -tanh
                                                            kind of above
                                                        -vary
                                                            kind of above
                                                        -Winners
                                                            "Acivation function _ winner.plk"
                                                            relu_winner.plk
                                                            ...
                                                            ...
                                                            vary_winner.plk
                                                    """
    global X_train, X_test, y_train, y_test, feature_names, target_names
    base = 'NEAT'
    config_name = 'config-recurrent'
    scores = ut.files_tour(X_train,X_test,y_train,y_test,base, 10, config_name,
                            feats=feature_names, targets=target_names, ret_check=0, replay='relu', rerun=1)


def fit_and_test_winner():
    global X_train, X_test, y_train, y_test, feature_names, target_names

    # An old winner that has 3 types of activation function.
    vary = ut.bring_NEAT_model(X_train,X_test,y_train,y_test, 'NEAT/config-recurrent', 'NEAT/Winners/vary_winner.plk', 'vary',
                         act_functions = ['relu','sigmoid','tanh','vary'])
    print(vary)  

    # An old winner that only use one type of activation function ('sigmoid') in all the network.
    sigmoid = ut.bring_NEAT_model(X_train,X_test,y_train,y_test, 'NEAT/config-recurrent', 'NEAT/Winners/sigmoid_winner.plk', 'sigmoid')
    print(sigmoid)


if __name__ == "__main__":
    global X_train, X_test, y_train, y_test, feature_names, target_names
    examples = [1,2,3]
    print(sys.argv[1])
    assert int(sys.argv[1]) in examples

    print('Importing Iris dataset')
    iris = load_iris()
    X = iris.data[:]
    y = iris.target

    sss = StratifiedShuffleSplit(n_splits = 1, test_size =0.3, random_state=1)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    feature_names = iris['feature_names']
    target_names = iris['target_names']

    print('Runnning example case',sys.argv[1])

    if int(sys.argv[1]) == 1:
        one_run()
    elif int(sys.argv[1]) == 2:
        new_evolution() # Warning: In the files_tour all the plot and save options are enabled
    elif int(sys.argv[1]) == 3:
        fit_and_test_winner()