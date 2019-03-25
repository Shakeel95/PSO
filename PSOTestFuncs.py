import numpy as np






############################ Contents ############################

# This contains all the test functions that are imported by PSOMain.py
# There are:
	# 1-Variable Functions: 1
	# 2-Variable Functions:	13
	# N-Variable Functions: 4







############################ Variable descriptions ############################

# loc is a list of the variable coordinates (aka location)








############################ 1-variable functions ############################
def xsq(loc):
    loc = np.array(loc)
    fout = loc[0] ** 2
    return fout







############################ 2-variable functions ############################
def booth(loc):
    loc = np.array(loc)
    fout = (loc[0] + 2 * loc[1] - 7) ** 2 + (2 * loc[0] + loc[1] - 5) ** 2
    return fout


def beale(loc):
    loc = np.array(loc)
    fout = (1.5 - loc[0] + loc[0] * loc[1]) ** 2 + (2.25 - loc[0] + loc[0] * loc[1] ** 2) ** 2 + (
                2.625 - loc[0] + loc[0] * loc[1] ** 3) ** 2
    return fout


def matyas(loc):
    loc = np.array(loc)
    fout = 0.26 * (loc[0] ** 2 + loc[1] ** 2) - 0.48 * loc[0] * loc[1]
    return fout


def threehumpcamel(loc):
    loc = np.array(loc)
    fout = 2 * loc[0] ** 2 - 1.05 * loc[0] ** 4 + (1 / 6) * loc[0] ** 6 + loc[0] * loc[1] + loc[1] ** 2
    return fout


def bukin_n6(loc):
    loc = np.array(loc)
    fout = 100 * np.sqrt(abs(loc[1] - 0.01 * loc[0] ** 2)) + 0.01 * abs(loc[1] + 10)
    return fout


def goldsteinprice(loc):
    loc = np.array(loc)
    fout = (1 + ((loc[0] + loc[1] + 1) ** 2) * (
                19 - 14 * loc[0] + 3 * loc[0] ** 2 - 14 * loc[1] + 6 * loc[0] * loc[1] + 3 * loc[1] ** 2)) * (
                       30 + ((2 * loc[0] - 3 * loc[1]) ** 2) * (
                           18 - 32 * loc[0] + 12 * loc[0] ** 2 + 48 * loc[1] - 36 * loc[0] * loc[1] + 27 * loc[1] ** 2))
    return fout


def levi_n13(loc):
    loc = np.array(loc)
    fout = (np.sin(3 * np.pi * loc[0])) ** 2 + ((loc[0] - 1) ** 2) * (1 + (np.sin(3 * np.pi * loc[1])) ** 2) + (
                (loc[1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * loc[1])) ** 2)
    return fout


def easom(loc):
    loc = np.array(loc)
    fout = -np.cos(loc[0]) * np.cos(loc[1]) * np.exp(-((loc[0] - np.pi) ** 2 + (loc[1] - np.pi) ** 2))
    return fout


def mccormick(loc):
    loc = np.array(loc)
    fout = np.sin(loc[0] + loc[1]) + (loc[0] - loc[1]) ** 2 - 1.5 * loc[0] + 2.5 * loc[1] + 1
    return fout


def ackley(loc):
    loc = np.array(loc)
    fout = -20 * np.exp(-0.2 * np.sqrt(0.5 * (loc[0] ** 2 + loc[1] ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * loc[0]) + np.cos(2 * np.pi * loc[1]))) + np.exp(1) + 20
    return fout


def eggholder(loc):
    loc = np.array(loc)
    fout = -(loc[1] + 47) * np.sin(np.sqrt(abs(0.5 * loc[0] + (loc[1] + 47)))) - loc[0] * np.sin(
        np.sqrt(abs(loc[0] - (loc[1] + 47))))
    return fout


def schaffer_n2(loc):
    loc = np.array(loc)
    fout = 0.5 + (((np.sin(loc[0] ** 2 - loc[1] ** 2)) ** 2 - 0.5) / ((1 + 0.001 * (loc[0] ** 2 + loc[1] ** 2)) ** 2))
    return fout


def schaffer_n4(loc):
    loc = np.array(loc)
    fout = 0.5 + (((np.cos(np.sin(abs(loc[0] ** 2 - loc[1] ** 2)))) ** 2 - 0.5) / (
                (1 + 0.001 * (loc[0] ** 2 + loc[1] ** 2)) ** 2))
    return fout










############################ N-variable functions ############################

def sphere(loc):
    loc = np.array(loc)
    fout = sum(loc ** 2)
    return fout


def rosenbrock(loc):
    loc = np.array(loc)
    n = len(loc)
    xi = loc[0:(n - 1)]
    xiplus1 = loc[1:n]
    fout = sum(100 * (xiplus1 - xi) ** 2 + (1 - xi) ** 2)
    return fout


def Styblinski_Tang(loc):
    loc = np.array(loc)
    fout = sum(loc ** 4 - 16 * loc ** 2 + 5 * loc) / 2
    return fout


def rastrigin(loc):
    loc = np.array(loc)
    n = len(loc)
    fout = 10 * n + sum(loc ** 2 - 10 * np.cos(2 * np.pi * loc))
    return fout
