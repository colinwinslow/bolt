'''
Created on Jul 5, 2012

@author: colinwinslow
'''

#tiny change2
 
import numpy as np

from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import cluster_util
from planar import Vec2,BoundingBox
import landmark


def clusterCostWorker(data,objectDict,maxDiscountRatio,strong=False):
    #maxDiscountRatio is the maximum amount of the groups cost that can be deducted
    #for membership in a group. Higher values will make groups more favorable compared
    # to lines.
    output = []
    for i in data:
        clusterMembers = map(lambda x: objectDict.get(x),i)
        hullAroundMembers = cluster_util.convex_hull(clusterMembers)
        areaOfMembers = sum(cluster_util.bb_area(objectDict.get(p).representation.rect) for p in i)
        try:
            density = areaOfMembers/cluster_util.area(hullAroundMembers)
        except:
            density = 0
        if density>=1: density = 1
        cost = len(i)*(1-density**2)*maxDiscountRatio + len(i)*(1-maxDiscountRatio)
        
#        corecluster=cluster_util.GroupBundle(i,cost,hullAroundMembers)
 
        corecluster=landmark.GroupRectangleRepresentation([objectDict.get(j) for j in i],None,cost)
        corecluster.density = density
        corecluster.isStrong = strong
        output.append(corecluster)
    return output
def clustercost(data,objectDict):
    #    data is a tuple of two dictionaries: core cluster data first, then larger and more permissive clusters\
    # this func needs to return a unified list of possible clusters using both dictionaries in the style of the chain finder function
    # quick and dirty:
    
    smallClusters = clusterCostWorker(data[0].values(),objectDict, 0.2,1)
    bigClusters =  clusterCostWorker(data[1].values(),objectDict, 0.1,0)
        
    return (smallClusters,bigClusters)




def dbscan(data,distanceMatrix,objectDict):
    X = [o.uuid for o in data]
    S = 1 - (distanceMatrix / np.max(distanceMatrix))
    db = DBSCAN(min_samples=4).fit(S)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    clusterlist = zip(labels, X)
    shortclusterlist = zip(labels,X)


    fringedict = dict()
    coredict = dict()

    for i in core_samples:
        ikey = int(clusterlist[i][0])
        ival = clusterlist[i][1]
        try:
            coredict[ikey].append(ival)
            fringedict[ikey].append(ival)
        except:
            coredict[ikey]=[]
            fringedict[ikey]=[]
            coredict[ikey].append(ival)
            fringedict[ikey].append(ival)
#        print clusterlist[i]
        shortclusterlist.remove(clusterlist[i])

    for i in shortclusterlist:
        try:
            fringedict[int(i[0])].append(i[1])
        except:
            fringedict[int(i[0])]=[]
            fringedict[int(i[0])].append(i[1])
            
    return (coredict,fringedict)
