import numpy as np


def posit_init(bound, s):
    return np.random.uniform(bound[0], bound[1], s)


def veloc_init(bound, s):
    prange = bound[1] - bound[0]
    return np.random.uniform(-abs(prange), abs(prange), s)



def pso_init(f, s, bounds):
    n = len(bounds)

    # particle current positions and velocities
    pcurr = list(map(posit_init, bounds, [s] * n))
    pcurr = np.array(list(map(list, zip(*pcurr))))
    vcurr = list(map(veloc_init, bounds, [s] * n))
    vcurr = np.array(list(map(list, zip(*vcurr))))

    # particle best position and value
    pbest = np.copy(pcurr, order="k")
    fbest = np.array(list(map(f, pbest)))

    # global best position and valueabs
    fgbest = min(fbest)
    pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

    return pcurr, vcurr, pbest, fbest, pgbest, fgbest




def qpso_init(f, s, bounds):
    n = len(bounds)

    # particle current positions and velocities
    pcurr = list(map(posit_init, bounds, [s] * n))
    pcurr = np.array(list(map(list, zip(*pcurr))))

    # particle best position and value
    pbest = np.copy(pcurr, order="k")
    fbest = np.array(list(map(f, pbest)))

    # global best position and valueabs
    fgbest = min(fbest)
    pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

    return pcurr, pbest, fbest, pgbest, fgbest
