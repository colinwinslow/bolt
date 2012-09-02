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


def clusterCostWorker(data,objectDict):
    output = []
    for i in data:
        corePO = map(lambda x: objectDict.get(x),i)
        coreHull = cluster_util.convex_hull(corePO)
        area = sum(objectDict.get(p).bb_area() for p in i)
        try:
            density = area/cluster_util.area(coreHull)
        except:
            density = 0
        if density > 1.3:
            groupAwesomeness = 1
        else: groupAwesomeness = density/1.3
        cost = .8*len(i)+.2*len(i) * np.sqrt(1-groupAwesomeness**2)
        
        corecluster=cluster_util.GroupBundle(i,cost)
        
        output.append(corecluster)
    return output
def clustercost(data,objectDict):
    #    data is a tuple of two dictionaries: core cluster data first, then larger and more permissive clusters\
    # this func needs to return a unified list of possible clusters using both dictionaries in the style of the chain finder function
    # quick and dirty:
    
    smallClusters = clusterCostWorker(data[0].values(),objectDict)
    bigClusters =  clusterCostWorker(data[1].values(),objectDict)
        
    return (smallClusters,bigClusters)




def dbscan(data,distanceMatrix,objectDict):
#    print "starting dbscan"
#    print "dbscan input:", data
    X,ids,bbmin,bbmax = zip(*data)
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
