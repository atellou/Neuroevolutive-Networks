
import pandas as pd
import numpy as np
import sklearn.metrics
import os
import neat
import pickle   
from neuroevo import fitness_func,run

'#######         Utility          ######'


def verify_in(vect, lookfor):
    """ Search within the data inside the vector for the "lookfor" """
    for i in vect:
        if lookfor in i:
            return True
    return False


def bring_last(vect, acti, split='-', pos=-1, func=None):
    """Designed to search based on the func criteria, it searches within the data in the vector
    split the data by "split", takes the last argument, and compares with the past argument.
    Designed to look for the last checkpoint."""
    if func == None:
        func = lambda a, b: float(a) > float(b)
    out = None
    ind = None
    for i in vect:
        dato = i.split(split)[pos]
        if (out == None or func(dato, out)) and (acti in i):
            out = dato
            ind = i
    return out, ind

def modify_config(config, activation_def, fitnes_th, n_inputs, n_outputs, activation_opt=None, n_hidden=None):
    """
    This function was created to modify the config parameters somewhat automatically
    :param config: (Object or Str) Configuration file from NEAT, could be the objecto or the path
    :param activation_def: (Str) Activation function of the first layer
    :param fitnes_th: (Int or float) Fitness_threshold
    :param n_inputs: (Int) number of input nodes
    :param n_outputs: (Int) number of output nodes
    :param activation_opt: (Vactor of strings)  The possible options of activation function for the hidden and output layers
    :param n_hidden: (Int) Number of nodes to add to each genome in the initial population
    :return: Config object
    """
    feats = [-1 * f for f in range(1, n_inputs + 1)]
    targets = [t for t in range(n_outputs)]
    if type(config) == str:
        print('Configuration file:', config)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config)
    '[NEAT]'
    config.fitness_threshold = fitnes_th
    '[DeafaultGenome]'
    config.genome_config.activation_default = activation_def
    if activation_opt == None:
        config.genome_config.activation_options = [activation_def]
    else:
        config.genome_config.activation_options = activation_opt
    # Network Parameters
    if n_hidden != None:
        config.genome_config.num_hidden = n_hidden

    config.genome_config.num_inputs = n_inputs
    config.genome_config.input_keys = feats

    config.genome_config.num_outputs = n_outputs
    config.genome_config.output_keys = targets

    return config


def files_tour(X_train, X_test, y_train, y_test, base, generations, config_name,fitness_metric,
    CV=None, feats=[], targets=[], ret_check=1, replay=None, rerun=0, verbose=0, view=0):
    """
    :param X_... or y_...: (array) Data to train and test
    :param base: Path to save all
    :param feats: (array) Features names
    :param targets: (array) Target names
    :param base: (Str) Folder that contains all the hierarchy necessary to this example function
    :param generations: (int) How many generation to run
    :param fitness_metric: (function) Function to compute the fitness level, it has to recive two vector parameters, targets and predictions in that order.
    :param CV: (Vector) If passed, then kind of cross_validation will be performed, it has to be a vector of three positions = [n_splits, random_state, shufle]
                It use the function "sklearn.model_selection.StratifiedKFold()"
    :param ret_checks: (Boolean) If True returns to the most recent checkpoint, otherwise start from cero
    :param replay: (Str or None) If type == Str and == to the activation_function then despite exist a winner,
    It will run again.
    :param rerun: (Boolean) If True it will ignore if alredy exist a winner for all the activation functions between the options
    :return: A Dataframe containing the results obtained"""
    
    scores = {'columns': ['fitness', 'accuracy', 'f1_score', 'recall', 'presicion', 'kappa']}
    act_functions = ['relu', 'sigmoid', 'tanh', 'vary']
    save_im = 'Images'
    save_ch = 'Check'
    save_wn = 'Winners'
    
    row, col = np.shape(X_train)  # No. Samples and Features
    n_targets = len(list(dict.fromkeys(y_train)))  # No. targets

    win_path = os.path.join(base, save_wn)  # general path + Winners
    winners_store = os.listdir(win_path)  # What is inside the winners path?
    for act in act_functions:
        ch_path = os.path.join(base, act, save_ch)  # general + activation function + CheckPoints
        ch_store = os.listdir(ch_path)  # What is inside the checkpoints file
        if not (verify_in(winners_store, act)) or (
                act == replay) or rerun:  # Do not enter If already exist a winner for the current "act" unless "rerun" == True
            if verify_in(ch_store, act) and ret_check:  # si hay un check point
                return_check = os.path.join(ch_path, bring_last(ch_store, act)[-1])
            else:
                return_check = None
            if act == 'vary':
                config = modify_config(os.path.join(base, config_name),  # crate the config file
                                       activation_def='relu', fitnes_th=1, n_inputs=col, n_outputs=n_targets,
                                       activation_opt=act_functions[:-1])
            else:
                config = modify_config(os.path.join(base, config_name),  # crate the config file
                                       activation_def=act, fitnes_th=1, n_inputs=col, n_outputs=n_targets)
            print('Configuration parameters:'
                  '\nActivation function:', config.genome_config.activation_default,
                  '\nActivation options:', config.genome_config.activation_options,
                  '\nNumber of entries:', config.genome_config.num_inputs,
                  '\nNumber of outputs:', config.genome_config.num_outputs,
                  '\nfilename_prefix=', os.path.join(ch_path, act + '_neat-checkpoint-'),
                  '\nfilename_fitness=', os.path.join(base, act, save_im, act + '_avg_fitness.svg'),
                  '\nfilename_spec=', os.path.join(base, act, save_im, act + '_speciation.svg'),
                  '\nfilename_net=', os.path.join(base, act, save_im, act + '_net_graph'),
                  '\nfilename_best=', os.path.join(win_path, act + '_winner'))
            fitness = fitness_func(X_train,y_train,
                                    fitness_metric, CV)  # create the object to store the train data and pass the fitness function
            winner, predictions, stats = run(config,
                                             X_test, y_test, fitness,
                                             generations=generations,
                                             filename_prefix=os.path.join(ch_path, act + '_neat-checkpoint-'),
                                             filename_fitness=os.path.join(base, act, save_im,
                                                                           act + '_avg_fitness.svg'),
                                             filename_spec=os.path.join(base, act, save_im, act + '_speciation.svg'),
                                             filename_net=os.path.join(base, act, save_im, act + '_net_graph'),
                                             filename_best=os.path.join(win_path, act + '_winner'),
                                             return_check=return_check,
                                             save_check=1, save=1, save_best=1,
                                             verbose=verbose, view=view, feats=feats, targets=targets)
            pred_targets = np.argmax(predictions, axis=1)
            accuracy = sklearn.metrics.accuracy_score(y_test, pred_targets)
            recall = sklearn.metrics.recall_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
            f1_score = sklearn.metrics.f1_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
            presicion = sklearn.metrics.precision_score(y_test, pred_targets, labels=range(n_targets),
                                                        average='weighted')
            kappa = sklearn.metrics.cohen_kappa_score(y_test, pred_targets, labels=range(n_targets))
            scores[act] = [winner.fitness,accuracy, f1_score, recall, presicion, kappa]
            print('Termine ', os.path.join(base, act))
            pd.DataFrame(scores, index=scores['columns']).drop(['columns'], axis=1).to_csv(
                os.path.join(base, 'metrics.csv'))
    return pd.DataFrame(scores, index=scores['columns']).drop(['columns'], axis=1).T


def replay_genome(genome_path):
    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    return genome

def bring_NEAT_model(X_train, X_test, y_train, y_test, config_path, model_path, act,
                     act_functions=['relu', 'sigmoid', 'tanh', 'vary']):
    """
    :param X_... or y_...: (array) Data to train and test
    :param config_path: (Object or Str) Configuration file from NEAT, could be the objecto or the path
    :param model_path: (str) Path where the NEAT model is
    :param act (Str): activation function to use in the model, if 'vary' passed then it will use act_functions
                      starting by 'relu' and mutating it with a proabability of 20% by default.
    :param act_functions: list of activation functions to use in case that act=='vary'
    """
    
    '##### Bring data  ######'
    row, col = np.shape(X_train)
    n_targets = len(list(dict.fromkeys(y_train)))
    '##### Bring model ######'
    if 'vary' in model_path:
        config = modify_config(config_path,  # crate the config file
                               activation_def='relu', fitnes_th=1, n_inputs=col, n_outputs=n_targets,
                               activation_opt=act_functions[:-1])
    else:
        config = modify_config(config_path,  # crate the config file
                               activation_def=act, fitnes_th=1, n_inputs=col, n_outputs=n_targets)
    # config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
    # neat.DefaultSpeciesSet, neat.DefaultStagnation,
    # config_path)
    winner = replay_genome(model_path)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    prediction = [winner_net.activate(xi) for xi in X_test]
    scores = {}
    pred_targets = np.argmax(prediction, axis=1)
    scores['accuracy'] = sklearn.metrics.accuracy_score(y_test, pred_targets)
    scores['recall'] = sklearn.metrics.recall_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
    scores['f1_score'] = sklearn.metrics.f1_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
    scores['precision'] = sklearn.metrics.precision_score(y_test, pred_targets, labels=range(n_targets), average='weighted')
    scores['kappa'] = sklearn.metrics.cohen_kappa_score(y_test, pred_targets, labels=range(n_targets))
    scores['c_matrix'] = sklearn.metrics.confusion_matrix(y_test, pred_targets, labels=range(n_targets), normalize='true')
    return winner, winner_net, y_test, pred_targets, scores