This repository contains some useful functions to run a model based on the Neuroevolutive principle. These functions, for the moment, are based on the NEAT library of python.

* The neuroevo.py file contains a class object to define the fitness function. Additionally, a modified version of the run script gave by NEAT.
* The Visualize and config files are the scripts also given by NEAT to run the code. The Visualize file contains some slight modifications.
* The script utility.py contains some functions designed to run the code and use **run** function in neuroevo.py.

In the main.py script are two example cases to use som function in utility.py


Some results example case 2:
======================

Terminal:
----------------------
```
$ python3 path\main.py 2
```

Results:
----------------------

If the generation of the images is available, the following plots will appear:
Fitness results for a 10 generation run of Recurrent Neural Network:
![alt text](https://github.com/atellou/Neuroevolutive-Networks/blob/main/NEAT/relu/Images/relu_avg_fitness.svg.svg)

Graph of the network:
![alt text](https://github.com/atellou/Neuroevolutive-Networks/blob/main/NEAT/relu/Images/relu_net_graph.svg)

Graph of the speciation:
![alt text](https://github.com/atellou/Neuroevolutive-Networks/blob/main/NEAT/relu/Images/relu_speciation.svg.svg)


Additionally, the scores and winners will remain in the folder NEAT

