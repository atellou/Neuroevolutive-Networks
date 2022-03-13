import numpy as np
import pandas as pd
import sklearn.metrics
import os
import sys

sys.path.append('C:\\Users\\andre\\Documents\\pythonProject\\NeuroEvo\\Neuroevolutiv-Networks')

import neuroevo as ne
import utility as ut

X_train = np.array(pd.read_csv('C:/Users/andre/Documents/pythonProject/Neuroevo_Ansiedad/X_train.csv').values)
X_test = np.array(pd.read_csv('C:/Users/andre/Documents/pythonProject/Neuroevo_Ansiedad/X_test.csv').values)
y_test = np.array(pd.read_csv('C:/Users/andre/Documents/pythonProject/Neuroevo_Ansiedad/y_test.csv').values).T[0]
y_train = np.array(pd.read_csv('C:/Users/andre/Documents/pythonProject/Neuroevo_Ansiedad/y_train.csv').values).T[0]

row, col = np.shape(X_train)  # No. Samples and Features
n_targets = len(list(dict.fromkeys(y_train)))  # No. targets

base = 'C:/Users/andre/Documents/pythonProject/Neuroevo_Ansiedad/NEAT2'
config_name = 'config-recurrent'
act = 'relu'
scores = {'columns': ['fitness', 'f1_score', 'recall', 'presicion', 'kappa']}
CV = [3, 1, True] #n_splits, random_state, shufle 

population =  ut.replay_genome('relu_neat-checkpoint-1103.plk')

def metric(targets, predictions):
    confusion_mat = sklearn.metrics.confusion_matrix(targets, predictions,
                                                    labels=range(len(list(dict.fromkeys(targets)))), normalize='true')
    diagonal_avg = np.mean(np.diagonal(confusion_mat))
    f1 = sklearn.metrics.f1_score(targets, predictions,labels=range(len(list(dict.fromkeys(targets)))), average='weighted')
    return (f1*0.6)+(diagonal_avg*0.4)

config = ut.modify_config(os.path.join(base, config_name),  # crate the config file
                            activation_def=act, fitnes_th=1, n_inputs=col, n_outputs=n_targets)
fitness = ne.fitness_func(X_train,y_train,
                                metric, CV)  # create the object to store the train data and pass the fitness function
winner, predictions, stats = ne.run(config,
                                X_test, y_test, fitness,
                                generations=3000, check_interval=500, check_time=1800,
                                filename_prefix=os.path.join(base,act,'Check' ,act + '_neat-checkpoint-'),
                                filename_fitness=os.path.join(base, act, 'Images',
                                                            act + '_avg_fitness.svg'),
                                filename_spec=os.path.join(base, act, 'Images', act + '_speciation.svg'),
                                filename_net=os.path.join(base, act, 'Images', act + '_net_graph'),
                                filename_best=os.path.join(base,'Winners', act + '_winner'),
                                run_again=0,
                                return_check = population,
                                save_check=1, save=1, save_best=1,
                                verbose=1, view=1)
pred_targets = np.argmax(predictions, axis=1)
recall = sklearn.metrics.recall_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
f1_score = sklearn.metrics.f1_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
presicion = sklearn.metrics.precision_score(y_test, pred_targets, labels=range(n_targets),
                                            average='weighted')
kappa = sklearn.metrics.cohen_kappa_score(y_test, pred_targets, labels=range(n_targets))
scores[act] = [winner.fitness, f1_score, recall, presicion, kappa]
print('Termine ', os.path.join(base, act))
print(pd.DataFrame(scores, index=scores['columns']).drop(['columns'], axis=1).T)