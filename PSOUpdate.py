import numpy as np




def veloc_update(pcurr, vcurr, pbest, pgbest, params):
    rpg = np.random.uniform(0, 1, 2)
    vcurr = params[0] * vcurr + params[1] * rpg[0] * (pbest - pcurr) + params[2] * rpg[1] * (pgbest - pcurr)
    return vcurr


def posit_update(pcurr, vcurr, bounds):
    newp = pcurr + vcurr
    for i in range(len(newp)):
        if newp[i] > bounds[0] and newp[i] < bounds[1]:
            pcurr[i] = newp[i]
    return pcurr



def point_update(pcurr, vcurr, pbest, fbest, pgbest, params, bounds, f):
    n = len(pcurr)
    for d in range(n):
        vcurr[d] = veloc_update(pcurr[d], vcurr[d], pbest[d], pgbest[d], params)
    newp = pcurr + vcurr
    for d in range(n):
        if newp[d] > bounds[d][0] and newp[d] < bounds[d][1]:
            pcurr[d] = newp[d] + 0
    fcurr = f(pcurr)
    if fcurr < fbest:
        fbest = fcurr + 0
        pbest = pcurr + 0
    return pcurr.tolist(), vcurr.tolist(), pbest.tolist(), fbest


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
            x[d] = newx_id + 0
    fcurr = f(x)
    if fcurr < fbest:
        fbest = fcurr + 0
        pbest = x + 0
    return x.tolist(), pcurr.tolist(), pbest.tolist(), fbest



def dist(v1, v2):
    distt = np.sum(np.sqrt(np.sum(np.square(v1 - v2), axis=1)))
    return distt
