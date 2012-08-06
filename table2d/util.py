from collections import namedtuple
#from bwdb import *
import numpy as np

import math


def totuple(a):
    '''converts whatever nested iterables into tuples'''
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
    
    
    
def findDistance(vector1,vector2):
    '''euclidian distance between 2 points'''
    return np.round(math.sqrt(np.sum(np.square(vector2-vector1))),3) 

def findAngle(vector1, vector2):
    '''difference between two vectors, in radians'''
    preArc=np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)
    return np.arccos(np.round(preArc, 5))

def find_pairs(l):
    '''returns a list of all possible pairings of list elements'''
    pairs = []
    for i in range(len(l)):
        for j in range(len(l))[i + 1:]:
            pairs.append((l[i], l[j]))
    return pairs

