' ### Libraries ####'

import pandas as pd
import numpy as np
import sklearn
import visualize
import pickle   
import neat
import uuid
import os
from random import randint
'####### Neuroevolutive algorithm ######'

class fitness_func(object):
    '''
    :param train_x or train_y: (array) Data to train and test
    :param fitness_metric: (function) Function to compute the fitness level, it has to recive two vector parameters, targets and predictions in that order.
    :param CV: (Vector) If passed, then kind of cross_validation will be performed, it has to be a vector of three positions = [n_splits, random_state, shufle]
                It use the function "sklearn.model_selection.StratifiedKFold()"
    :return: (Object) This object contains the functions plus the data needed to compute fitness function necessary for the neat algorithm.
    '''
    def __init__(self, train_x, train_y, fit_metric, CV=None):
        self.train_x, self.train_y = train_x, train_y
        self.metric = fit_metric
        if CV!=None:
            self.cross_val(CV[0],CV[1],CV[2])
            self.method = lambda genomes, config: self.eval_fitness(genomes, config, self.skf[randint(0,CV[0]-1)][0])
        else:
            self.method = lambda genomes, config: self.eval_fitness(genomes, config, range(0,len(train_y)))

    def eval_fitness(self, genomes, config, train_index):
        '''Apply the metric passed to all the genomes and compute each fitness'''
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            predictions = [net.activate(x) for x in self.train_x[train_index]]
            genome.fitness = self.metric(self.train_y[train_index], np.argmax(predictions, axis=1))

    def cross_val(self, n_splits, random_state, shufle):
        '''Apply the StratifiedKFold from sklearn and create a new attribute with those folders'''
        skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shufle)
        self.skf = list(skf.split(self.train_x, self.train_y)) #[(train_index, test_index), ...]

    def fitness(self, genome, config):
        '''Compute the fitness function'''
        self.method(genome, config)


def run(config, test_x, test_y, fitness,
        generations=1000, generation_interval=600,
        filename_prefix='./neat-checkpoint-',
        filename_fitness='./avg_fitness', filename_spec='./speciation',
        filename_net='./net_graph', filename_best='./better',
        run_again=0, return_check=None,
        verbose=False, view=0, save_check=0, save=0, save_best=0,
        feats=[], targets=[]):
    '''
    :param config: (str or object) Configuration file from NEAT, could be the objecto or the path
    :param train_y: (np.array) Labels for train set
    :param test_x: (np.array) Data for test set
    :param test_y: (np.array) Labels for test set
    :param fitness: (Object) Object with Train data and fitness function
    :param generations: (int) How many generation to run
    :param generation_interval: (int) Interval in seconds to create a chack point
    :param filename_prefix: (Str) Path to save the check points
    :param filename_fitness: (Str) Path to save the fitness progration, save an image
    :param filename_spec: (Str) Path to save an image with the changes in the population
    :param filename_net: (Str) Path to save an image with a scheme of the winner net
    :param filename_best: (Str) Path to save the winner
    :param run_again: (Boolean) If a check point was given, if False only the remaining generation to complete "generation"
    will run, if True, it computes all the "generations" specified
    :param return_check: (Str or None) If None, the process starts from zero, otherwise, the path specified is taken as
    the check point to start with
    :param verbose: (Boolean) To have a feedback of the process
    :param view:  (Boolean) If plot the graphs
    :param save_check: (Boolean) If make check points
    :param save: (Boolean) If save the images
    :param save_best: (Boolean) If save the winner
    :param feats: Vector Strings with feats names to plot the Neural network
    :param targets: Vector Strings of target names to plot the Neural network
    :return: The winner, the predictions made for Test set, and group of stats given by NEAT
    '''

    # feats
    def crate_nodes(feats, targets, test_x, test_y):
        # Only used for visualization
        node_names = {}
        if feats == []:
            n_feats = np.shape(test_x)[-1]
            feats = ['feat' + str(f) for f in range(n_feats)]
        if targets == []:
            n_targets = len(list(dict.fromkeys(test_y)))
            targets = ['target' + str(t) for t in range(n_targets)]
        for i, f in enumerate(feats):
            node_names[-1 * (i + 1)] = f
            # print(-1*(i+1), f)
        for i, f in enumerate(targets):
            node_names[i] = f
        return node_names

    # Load configuration.
    if type(config) == str:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config)

    if return_check == None:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        if verbose:
            p.add_reporter(neat.StdOutReporter(verbose))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        if save_check:
            p.add_reporter(neat.Checkpointer(generation_interval,
                                             filename_prefix=filename_prefix))

        # Run for up to 1000 generations.
        winner = p.run(fitness.fitness, generations)
    else:
        print(
            '\n=================================== Return to ' + return_check + ' ===================================\n')
        last_generation = int(return_check.split('-')[-1])
        p = neat.Checkpointer.restore_checkpoint(return_check)
        if verbose:
            p.add_reporter(neat.StdOutReporter(verbose))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        if save_check:
            p.add_reporter(neat.Checkpointer(generation_interval,
                                             filename_prefix=filename_prefix))
        if run_again:
            winner = p.run(fitness.fitness, generations)
        else:
            print('\n             ====================           ' + str(
                generations - last_generation) + ' generations  to go      ====================      \n')
            winner = p.run(fitness.fitness, generations - last_generation)

    # Display the winning genome.
    print('===================================' * 3)
    # print('\nMaximum possible fitness value:\n{!s}'.format(np.shape(train_y)[0]))
    print('\nBest genome:\n{!s}'.format(winner).split('\nConnections:\n\tDefaultConnectionGene')[0])

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    prediction = [winner_net.activate(xi) for xi in test_x]

    if save_best:
        with open(filename_best + '.plk', "wb") as f:
            pickle.dump(winner, f)
            f.close()
    node_names = crate_nodes(feats, targets, test_x, test_y)
    if not (save):
        name_net = str(uuid.uuid4())
        visualize.draw_net(config, winner, view=view, node_names=node_names, filename= name_net)
        visualize.plot_stats(stats, ylog=False, view=view, filename=filename_fitness + '.svg')
        visualize.plot_species(stats, view=view, filename=filename_spec + '.svg')

        os.remove(name_net)
        os.remove(name_net+'.svg')
        return winner, prediction, stats

    print('save images')
    visualize.draw_net(config, winner, view=view, node_names=node_names, filename=filename_net)
    visualize.plot_stats(stats, ylog=False, view=view, filename=filename_fitness + '.svg', save=save)
    visualize.plot_species(stats, view=view, filename=filename_spec + '.svg', save=save)

    return winner, prediction, stats