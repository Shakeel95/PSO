import numpy as np








############################ Contents ############################

# This contains the two initialization functions that are called by PSOMain.py
# This also contains two initialization functions specifically for velocity and 
# 	position which are called by the two main initialization functions














############################ Variable descriptions ############################

# n is the number of dimensions (int)
# s is the number of particles (int)
# bounds of the search area - of the form [[x1min, x1max], ... , [xnmin, xnmax]]
# bound is the search area bound for one dimension (one entry of bounds)
# f is the function to be optimized

# pcurr is the current position of each particles
# vcurr is the current velocity of each particles
# pbest is the best position of each particle
# fbest is the minimum value found for each particle
# fgbest is the overall minimum value found of all particles
# pgbest is the overall best position of each particle












############################ Initialization Functions ############################




# Takes in bound for one dimension, and the number of particles (s)
# Randomly generations the position in that dimension of each particle based on 
# 	a uniform distribution within the bounds for the dimension

def posit_init(bound, s):
    return np.random.uniform(bound[0], bound[1], s)








# Takes in bound for one dimension, and the number of particles (s)
# Randomly generations the velocity in that dimension of each particle based on 
# 	a uniform distribution between the range times -1 and the range itself

def veloc_init(bound, s):
    dimrange = abs(bound[1] - bound[0])
    return np.random.uniform(-dimrange, dimrange, s)








# Called by PSOMain.py
# Takes in the function, the number of particles (s), and the bounds
# Uses these to initialize all necessary starting values for PSO

def pso_init(f, s, bounds):
    n = len(bounds)

    # initializes current particle positions and velocities using functions above
    pcurr = list(map(posit_init, bounds, [s] * n))
    pcurr = np.array(list(map(list, zip(*pcurr))))
    vcurr = list(map(veloc_init, bounds, [s] * n))
    vcurr = np.array(list(map(list, zip(*vcurr))))

    # best particle position and value automatically starts the same as current
    pbest = np.copy(pcurr, order="k")
    fbest = np.array(list(map(f, pbest)))

    # global best position and value are calculated
    fgbest = min(fbest)
    pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

    return pcurr, vcurr, pbest, fbest, pgbest, fgbest








# Called by PSOMain.py
# Takes in the function, the number of particles (s), and the bounds
# Uses these to initialize all necessary starting values for QPSO

def qpso_init(f, s, bounds):
    n = len(bounds)

    # initializes current particle positions using function above
    pcurr = list(map(posit_init, bounds, [s] * n))
    pcurr = np.array(list(map(list, zip(*pcurr))))

    # best particle position and value automatically starts the same as current
    pbest = np.copy(pcurr, order="k")
    fbest = np.array(list(map(f, pbest)))

    # global best position and value are calculated
    fgbest = min(fbest)
    pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

    return pcurr, pbest, fbest, pgbest, fgbest
