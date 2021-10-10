'''https://github.com/fmfn/BayesianOptimization'''
# Support for maths
import numpy as np
# Plotting tools
from matplotlib import pyplot as plt
# we use the following for plotting figures in jupyter
# GPy: Gaussian processes library
import GPy

from bayes_opt import BayesianOptimization



# Bounded region of parameter space


# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}





def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.
    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    
    return -x ** 2 - (y - 1) ** 2 + 1


optimizer = BayesianOptimization(
f=black_box_function,
pbounds=pbounds,
verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
random_state=1,
)


optimizer.maximize(
    init_points=2,
    n_iter=4,
)

print(optimizer.max['params'])