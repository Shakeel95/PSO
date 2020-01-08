import numpy as np








############################ Contents ############################

# This contains the three update functions that are called by PSOMain.py















############################ Variable descriptions ############################

# n is the number of dimensions (int)
# s is the number of particles (int)
# bounds of the search area - of the form [[x1min, x1max], ... , [xnmin, xnmax]]
# bound is the search area bound for one dimension (one entry of bounds)
# f is the function to be optimized
# params are the necessary parameters (omega, c1, c2) for PSO

# pcurr is the current position of each particles
# vcurr is the current velocity of each particles (PSO and parallelized PSO only)
# pbest is the best position of each particle
# fbest is the minimum value found for each particle
# fgbest is the overall minimum value found of all particles
# pgbest is the overall best position of each particle
# x
# mbest 

# newp is a temporary calculation of the new position, saved to pcurr if inside the bounds
# newx_id is a temporary calculation of the new x, saved to x if inside the bounds

# rpg
# phi
# u
# beta
# changeParam
# coinToss - 50% chance of being True / False, used in QPSO to decide to add or subtract changeParam






############################ Update Functions ############################



# Called by PSOMain.py (pso_algo())
# Takes in pcurr, vcurr, pbest for one dimension of one particle, as well as pgbest and params
# Updates the velocity based on the PSO algorithm

def veloc_update(pcurr, vcurr, pbest, pgbest, params):
    rpg = np.random.uniform(0, 1, 2)
    vcurr = params[0] * vcurr + params[1] * rpg[0] * (pbest - pcurr) + params[2] * rpg[1] * (pgbest - pcurr)
    return vcurr









# Called by PSOMain.py (pso_algo_par())
# Takes in pcurr, vcurr, pbest, fbest for one particle, as well as pgbest, params, bounds, f
# Performs the PSO algorithm to update pcurr, vcurr, pbest, and fbest

def point_update(pcurr, vcurr, pbest, fbest, pgbest, params, bounds, f):
    n = len(pcurr)
    for d in range(n):
        vcurr[d] = veloc_update(pcurr[d], vcurr[d], pbest[d], pgbest[d], params)
    newp = pcurr + vcurr
    for d in range(n):
        if newp[d] > bounds[d][0] and newp[d] < bounds[d][1]:
            #Adding 0 creates a new object in memory instead of variable that references same object
            pcurr[d] = newp[d] + 0
    fcurr = f(pcurr)
    if fcurr < fbest:
        fbest = fcurr + 0
        pbest = pcurr + 0
    return pcurr.tolist(), vcurr.tolist(), pbest.tolist(), fbest













# Called by PSOMain.py (qpso_algo_par())
# Takes in x, pcurr, pbest, fbest, for one particle, as well as mbest, pgbest, beta, bounds, f
# Performs the QPSO algorithm to update x, pcurr, pbest, and fbest

def qpoint_update(x, pcurr, pbest, fbest, mbest, pgbest, beta, bounds, f):
    n = len(x)
    for d in range(n):
        phi = np.random.uniform()
        u = np.random.uniform()
        coinToss = np.random.uniform() < 0.5
        pcurr[d] = phi * pbest[d] + (1 - phi) * pgbest[d]
        changeParam = beta * abs(mbest[d] - x[d]) * (-1) * np.log(u)
        newx_id = pcurr[d] + changeParam if coinToss else pcurr[d] - changeParam
        if newx_id > bounds[d][0] and newx_id < bounds[d][1]:
            #Adding 0 creates a new object in memory instead of variable that references same object
            x[d] = newx_id + 0
    fcurr = f(x)
    if fcurr < fbest:
        fbest = fcurr + 0
        pbest = x + 0
    return x.tolist(), pcurr.tolist(), pbest.tolist(), fbest
