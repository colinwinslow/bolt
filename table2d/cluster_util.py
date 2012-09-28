from collections import namedtuple
#from bwdb import *
import numpy as np
from uuid import uuid4
import math
from planar import BoundingBox,Vec2,Polygon
import landmark

    

def bb_area(bb):
    return bb.height*bb.width
    
ClusterParams = namedtuple("ClusterParams",['chain_distance_limit', 'angle_limit', 'min_line_length',
               'anglevar_weight', 'distvar_weight','dist_weight',
               'allow_intersection','beam_width','attempt_dnc'])



    
    
def create_distance_matrix(data):
    BoundingBoxes = [o.representation.rect for o in data]
    distance_array=[]
    for i in BoundingBoxes:
        row = []
        for j in BoundingBoxes:
            row.append(landmark.bb_to_bb_distance(i,j))
        distance_array.append(row)
    return distance_array

def convex_hull(data):
    BoundingBoxes = [o.representation.rect for o in data]
    polys = []
    for b in BoundingBoxes:
        polys.append(b.to_polygon())
    points = [] 
    for p in polys:
        output = [i for i in p]
        points = points + output
    hullpoly = Polygon.convex_hull(points)
    hullpoints = []
    for p in hullpoly:
        hullpoints.append(p)
    return hullpoints



def area(points):
    pivot = points[0]
    a = 0
    for i in range(len(points)-1):
        s1 = findDistance(pivot,points[i])
        s2 = findDistance(points[i],points[i+1])
        s3 = findDistance(points[i+1],pivot)
        s = 0.5 * ( s1 + s2 + s3)
        a = a + np.sqrt(s * (s - s1) * (s - s2) * (s - s3))
    return a

    
        
def convert_to_bboxes(data):
    BoundingBoxes = []
    for i in data:
        minvec = Vec2(i.bbmin[0],i.bbmin[1])
        maxvec = Vec2(i.bbmax[0],i.bbmax[1])
        BoundingBoxes.append(BoundingBox((minvec,maxvec)))
    return BoundingBoxes
        
def findDistance(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    '''euclidian distance between 2 points'''
    return np.sqrt(np.sum(np.square(vector2-vector1)))

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