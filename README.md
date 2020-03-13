# How it works

Sorry, it seems it was not properly uploaded the first time

This code works only for the class Approximant_NN, that is the equivalent to the simulation performed in the experiment
There are three main files

- classes/Approximant_NN: 
    - This class represents the function, and there are several functions
    - __init__: initialize some important parameters: domain, function, layers, parameters to encode the function
    - update_parameters: modify the parameters
    - run_complete: run the circuit for all points and creates the variables final_states or sampling, depending on whether noise is True or False
    - run: run the circuit for one batch and returns the results (noisy is also available)
    - _minim_function: compute chi
    - _minim_function_noisy: compute chi with sampling
    - find_optimal_parameters: this function performs the minimization. It calls functions in opt_algorithms. Now only minimize and adam_spsa_optimizer are functional
    - other functions are auxiliary functions
    
- opt_algorithms:
    - Several optimizations algorithms
    - Right now the only working algorithm is adam_spsa_optimizer (and auxiliary functions)

- opt:
    - Example file, contains example for minimization and testing of parameters