import numpy as np
import multiprocessing as mp
from itertools import repeat
import time
import pandas as pd


# Custom functions from other files we wrote
import PSOTestFuncs as tf
from PSOInit import pso_init
from PSOInit import qpso_init
from PSOUpdate import veloc_update
from PSOUpdate import point_update
from PSOUpdate import qpoint_update










############################ Contents ############################


# Defines the 4 algorithms that will be used:
#	- PSO - pso_algo()
#	- QPSO - qpso_algo()
#	- Parallelized PSO - pso_algo_par()
#	- Parallelized QPSO - qpso_algo_par()

# Runs 50 simulations of each algorithm on each test function within PSOTestFuncs.py
# Saves the output to a CSV

















############################ Variable descriptions ############################

# n is the number of dimensions (int)
# s is the number of particles (int)
# bounds of the search area - of the form [[x1min, x1max], ... , [xnmin, xnmax]]
# f is the function to be optimized
# params are the necessary parameters (omega, c1, c2) for PSO
# t is the current iteration of the algorithm
# sims is number of simulations to run on each function

# maxrounds is the maximum number of iterations allowed
# tol is the amount of change in fgbest to be considered an improvement
# nochange is the number or iterations without a sufficient improvement in fgbest before stopping
# samebest is a counter for how many rounds in a row with improvement in fgbest of less than tol

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
# fgbest_compare

# rpg
# phi
# u
# beta
# changeParam
# coinToss - 50% chance of being True / False, used in QPSO to decide to add or subtract changeParam














############################ Algorithm Functions ############################






# Takes in f, s, bounds, params, maxrounds, tol, nochange
# Runs PSO on f over the search area bounds using s particles and parameters params, 
#	and stopping criteria specified by maxrounds, tol, nochange
# Returns pgbest, fgbest, and t

def pso_algo(f, s, bounds, params, maxrounds, tol, nochange):
    n = len(bounds)
    pcurr, vcurr, pbest, fbest, pgbest, fgbest = pso_init(f, s, bounds)
    t = 0
    samebest = 0
    while t < maxrounds:
        fgbest_compare = fgbest
        for i in range(s):
            for d in range(n):
                vcurr[i][d] = veloc_update(pcurr[i][d], vcurr[i][d], pbest[i][d], pgbest[d], params)
            newp = pcurr[i] + vcurr[i]
            for d in range(n):
                if newp[d] > bounds[d][0] and newp[d] < bounds[d][1]:
                    #Adding 0 creates a new object in memory instead of variable that references same object
                    pcurr[i][d] = newp[d] + 0 
            fcurr = f(pcurr[i])
            if fcurr < fbest[i]:
                fbest[i] = fcurr + 0
                pbest[i] = pcurr[i] + 0
                if fcurr < fgbest:
                    fgbest = fcurr + 0
                    pgbest = pcurr[i] + 0
        t += 1

        if abs(fgbest_compare - fgbest) > tol :
            samebest = 0
        else :
            samebest += 1

        if samebest >= nochange :
            break
    return pgbest, fgbest, t

















# Takes in f, s, bounds, maxrounds, tol, nochange
# Runs QPSO on f over the search area bounds using s particles, 
#	and stopping criteria specified by maxrounds, tol, nochange
# Returns pgbest, fgbest, and t

def qpso_algo(f, s, bounds, maxrounds, tol, nochange):
    n = len(bounds)
    pcurr, pbest, fbest, pgbest, fgbest = qpso_init(f, s, bounds)
    x = np.copy(pcurr, order="k")
    t = 0
    samebest = 0
    while t < maxrounds:
        fgbest_compare = fgbest
        mbest = np.mean(pbest, axis=0)
        beta = 0.5*(maxrounds-t)/maxrounds + 0.5

        for i in range(s):
            for d in range(n):
                phi = np.random.uniform()
                u = np. random.uniform()
                coinToss = np.random.uniform() < 0.5
                pcurr[i,d] = phi*pbest[i,d] + (1- phi)*pgbest[d]
                changeParam = beta * abs(mbest[d] - x[i, d]) * (-1) * np.log(u)
                newx_id = pcurr[i, d] + changeParam if coinToss else pcurr[i, d] - changeParam
                if newx_id > bounds[d][0] and newx_id < bounds[d][1]:
                    #Adding 0 creates a new object in memory instead of variable that references same object
                    x[i,d] = newx_id + 0
            fcurr = f(x[i])
            if fcurr < fbest[i]:
                fbest[i] = fcurr + 0
                pbest[i] = x[i] + 0
                if fcurr < fgbest:
                    fgbest = fcurr + 0
                    pgbest = x[i] + 0
        t += 1

        if abs(fgbest_compare - fgbest) > tol:
            samebest = 0
        else:
            samebest += 1

        if samebest >= nochange:
            break
    return pgbest, fgbest, t














# Takes in f, s, bounds, params, maxrounds, tol, nochange
# Runs parallelized PSO on f over the search area bounds using s particles and parameters params, 
#	and stopping criteria specified by maxrounds, tol, nochange
# We update all the points in an iteration at once, so no communication within an iteration
# Returns pgbest, fgbest, and t

def pso_algo_par(f, s, bounds, params, maxrounds, tol, nochange):
    n = len(bounds)
    pcurr, vcurr, pbest, fbest, pgbest, fgbest = pso_init(f, s, bounds)
    t = 0
    samebest = 0
    while t < maxrounds:
        fgbest_compare = fgbest
        inputs = zip(pcurr, vcurr, pbest, fbest, repeat(pgbest), repeat(params), repeat(bounds), repeat(f))

        results_0 = pool.starmap(point_update, inputs)
        results = list(map(list, zip(*results_0)))

        pcurr = np.array(list(results)[0])
        vcurr = np.array(list(results)[1])
        pbest = np.array(list(results)[2])
        fbest = np.array(list(results)[3])

        if min(fbest) < fgbest:
            #Adding 0 creates a new object in memory instead of variable that references same object
            fgbest = min(fbest) + 0
            pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]
        t += 1

        if abs(fgbest_compare - fgbest) > tol:
            samebest = 0
        else:
            samebest += 1

        if samebest >= nochange:
            break
    return pgbest, fgbest, t




















# Takes in f, s, bounds, maxrounds, tol, nochange
# Runs parallelized QPSO on f over the search area bounds using s particles and parameters params, 
#	and stopping criteria specified by maxrounds, tol, nochange
# We update all the points in an iteration at once, so no communication within an iteration
# Returns pgbest, fgbest, and t

def qpso_algo_par(f, s, bounds, maxrounds, tol, nochange):
    pcurr, pbest, fbest, pgbest, fgbest = qpso_init(f, s, bounds)
    x = np.copy(pcurr, order="k")
    t = 0
    samebest = 0
    while t < maxrounds:
        fgbest_compare = fgbest
        mbest = np.mean(pbest, axis=0)
        beta = 0.5*(maxrounds-t)/maxrounds + 0.5

        inputs = zip(x, pcurr, pbest, fbest, repeat(mbest), repeat(pgbest), repeat(beta), repeat(bounds), repeat(f))

        results_0 = pool.starmap(qpoint_update, inputs)
        results = list(map(list, zip(*results_0)))

        x = np.array(list(results)[0])
        pcurr = np.array(list(results)[1])
        pbest = np.array(list(results)[2])
        fbest = np.array(list(results)[3])

        if min(fbest) < fgbest:
            #Adding 0 creates a new object in memory instead of variable that references same object
            fgbest = min(fbest) + 0
            pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

        t += 1

        if abs(fgbest_compare - fgbest) > tol:
            samebest = 0
        else:
            samebest += 1
        if samebest >= nochange:
            break
    return pgbest, fgbest, t




























############################ Simulations and Testing ############################



if __name__ == '__main__':


    # Specifies the necessary parameters to be used by the algorithms, and # of simulations

    s = 50
    params = [0.715, 1.7, 1.7]
    maxrounds = 1000
    tol = 10**(-9)
    nochange = 20
    sims = 50




    # Stores the information for each function including names of function as a string, 
    # how to call it, where the true minimum occurs, and what the bounds are

    funcnamelist = ["X-Squared", "Booth", "Beale", "ThreeHumpCamel", "GoldsteinPrice", "Levi_n13", "Sphere", "Rosebrock", "StyblinskiTang", "Ackley", "Schaffer_n2", "Eggholder", "McCormick", "Rastrigin", "Schaffer_n4", "Easom", "Bukin_n6", "Matyas"]
    functionlist = [tf.xsq, tf.booth, tf.beale, tf.threehumpcamel, tf.goldsteinprice, tf.levi_n13, tf.sphere, tf.rosenbrock, tf.Styblinski_Tang, tf.ackley, tf.schaffer_n2, tf.eggholder, tf.mccormick, tf.rastrigin, tf.schaffer_n4, tf.easom, tf.bukin_n6, tf.matyas]
    pminlist = [[0], [1,3], [3,0.5], [0,0], [0, -1],[1,1], [0,0,0,0], [1,1,1,1], [-2.903534,-2.903534,-2.903534,-2.903534,-2.903534,-2.903534], [0,0], [0,0], [512, 404.2319], [-0.54719, -1.54719], [0,0,0,0,0,0,0,0], [0,1.25313], [np.pi, np.pi], [-10,1], [0,0]]
    boundlist = [[[-200, 200]], [[-10, 10], [-10, 10]], [[-4.5, 4.5], [-4.5, 4.5]], [[-5, 5], [-5, 5]], [[-2, 2], [-2, 2]], [[-10, 10], [-10, 10]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-5, 5], [-5, 5]], [[-100, 100], [-100, 100]], [[-512, 512], [-512, 512]], [[-1.5, 4], [-3, 4]], [[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]], [[-100, 100], [-100, 100]], [[-100, 100], [-100, 100]], [[-15, -5], [-3, 3]], [[-10.00, 10.00], [-10.00, 9.00]]]




    # Sets up a dataframe to store the data 
    outdata = pd.DataFrame()




    # Sets up for parallel computing
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)









    # Forloop for each function containing for-loop for each simulation which runs all 4 algorithms and times them
    # Stores the results of each simulation and true function values in outdata, and saves this as a CSV

    for j in range(len(functionlist)):
        for k in range(sims):
            f = functionlist[j]
            bounds = boundlist[j]
            trueval = f(pminlist[j])

            start = time.time()
            pmin, fmin, nrounds = pso_algo(f, s, bounds, params, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, funcnamelist[j], "PSO", end-start, nrounds, pmin, pminlist[j], fmin, trueval]])

            start = time.time()
            pmin, fmin, nrounds = pso_algo_par(f, s, bounds, params, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, funcnamelist[j], "PSO_Par", end-start, nrounds, pmin, pminlist[j], fmin, trueval]])

            start = time.time()
            pmin, fmin, nrounds = qpso_algo(f, s, bounds, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, funcnamelist[j], "QPSO", end-start, nrounds, pmin, pminlist[j], fmin, trueval]])

            start = time.time()
            pmin, fmin, nrounds = qpso_algo_par(f, s, bounds, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, funcnamelist[j], "QPSO_Par", end-start, nrounds, pmin, pminlist[j], fmin, trueval]])

    pool.close()
    outdata.columns = ["Simulation#", "Function", "Method", "time", "rounds", "FoundMinLoc", "TrueMinLoc", "FoundMinVal", "TrueMinVal"]
    outdata.sort_values(["Function", "Method"], inplace = True)
    outdata = outdata.reset_index(drop = True)
    outdata.to_csv("OutputData.csv")