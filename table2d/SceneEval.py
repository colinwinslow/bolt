'''
Created on Jun 22, 2012
quick description and documentation in attached readme file. 
@author: colinwinslow
'''
import cluster_util
from cluster_util import ClusterParams
import numpy as np
import heapq
from cluster import dbscan,clustercost
from copy import copy
from time import time
import landmark

#vacuous comment for git practice



        
        

    

def sceneEval(inputObjectSet,params = ClusterParams(2,0.9,3,0.05,0.1,1,1,11,False)):
    
    '''
    find the clusters
    evaulate the inside of the clusters as lines to see if they'd be better as lines than clusters
    evaluate the outside of clusters for lines
    concatenate the lists of clusters and lines
    evaluate the whole thing with bundle search
    '''
    print "*",inputObjectSet
    reducedObjectSet = copy(inputObjectSet)
    objectDict = dict()
    for i in inputObjectSet:
        objectDict[i.uuid]=i
    distanceMatrix = cluster_util.create_distance_matrix(inputObjectSet)
    dbtimestart = time()
    clusterCandidates = clustercost(dbscan(inputObjectSet,distanceMatrix,objectDict),objectDict)
    dbtimestop = time()
    print "dbscan time: \t\t\t", dbtimestop-dbtimestart
    
    innerLines = []
    #search for lines inside large clusters
    if params.attempt_dnc==True:
        insideLineStart= time()
        for cluster in clusterCandidates[1]:

            innerObjects = []
            for id in cluster:
                for x in inputObjectSet:
                    if x.id == id:
                        innerObjects.append(x)
            innerChains = findChains(innerObjects,params)
            for thing in innerChains:
                innerLines.append(thing)
            
        #remove core clusters
        for cluster in clusterCandidates[0]:
            for id in cluster:
                for x in reducedObjectSet:
                    if x.id == id:
                        reducedObjectSet.remove(x)
        ReducedDistanceMatrix = cluster_util.create_distance_matrix(reducedObjectSet)
        insideLineStop = time()
        print "inside linesearch time:\t\t",insideLineStop-insideLineStart
        
    outsideLineStart = time()
    lineCandidates = findChains(reducedObjectSet,params,objectDict)
    outsideLineStop = time()
    
    print "general linesearch time:\t",outsideLineStop-outsideLineStart
    allCandidates = clusterCandidates[0]+clusterCandidates[1] + lineCandidates + innerLines
    allClusters = clusterCandidates[0]+clusterCandidates[1]
    allLines = lineCandidates + innerLines
    groupDictionary = dict()
    for i in allCandidates:
        groupDictionary[i.uuid]=i
    for i in inputObjectSet:
        groupDictionary[i.uuid]=i
    lineBundleStart = time()
    bestLines = bundleSearch(inputObjectSet, allLines, params.allow_intersection, params.beam_width)   
    lineBundleStop = time()
    print "linebundle time: \t\t", lineBundleStop-lineBundleStart
    
    clusterBundleStart = time()
    bestClusters = bundleSearch(inputObjectSet, allClusters, params.allow_intersection, params.beam_width) 
    print params.allow_intersection
    clusterBundleStop = time()
    print "clusterbundle time: \t\t", clusterBundleStop-clusterBundleStart
    bundleStart = time()
    evali = [] 

    try:
         evali = evali + bestLines
    except: print "there aren't any lines."
    try:
         evali = evali + bestClusters
    except: print "there aren't any clusterss."
    print "evali",evali
    
    
    output = map(lambda x: groupDictionary.get(x),evali)
    bundleStop = time()
    print "bundlesearch cleanup time: \t",bundleStop-bundleStart
    print "output",output
    return output

    

def findChains(inputObjectSet, params,objectDict):
    '''finds all the chains, then returns the ones that satisfy constraints, sorted from best to worst.'''

    bestlines = []
    explored = set()
    pairwise = cluster_util.find_pairs(inputObjectSet)
    pairwise.sort(key=lambda p: p[0].distance_to(p[1].representation),reverse=False)
    for pair in pairwise:
        start,finish = pair[0],pair[1]
        if frozenset([start.uuid,finish.uuid]) not in explored:
            result = chainSearch(start, finish, inputObjectSet,params)
            if result != None: 
                bestlines.append(result)
                s = map(frozenset,cluster_util.find_pairs(result[0:len(result)-1]))
                
                map(explored.add,s)

               
    verybest = []
    costSum = 0
    for line in bestlines:
        if len(line)>params.min_line_length:
            verybest.append(line)
    verybest.sort(key=lambda l: len(l),reverse=True)
    costs = map(lambda l: l.pop()+1.5,verybest)
    data = np.array(map(lambda x: (x.representation.middle,x.uuid),inputObjectSet))
    output = []
    for i in zip(costs,verybest):
        output.append(landmark.GroupLineRepresentation([objectDict.get(j) for j in i[1]],i[0]))
    return output
    
            
def chainSearch(start, finish, points, params):
    # Passing distancematrix in here to let us reuse it over and over for
    # calculating successor costs. Need to actually implement that, though. 
    node = Node(start, -1, [], 0,0)
    frontier = PriorityQueue()
    frontier.push(node, 0)
    explored = set()
    while frontier.isEmpty() == False:
        node = frontier.pop()
        if node.getState().uuid == finish.uuid:
            path = node.traceback()
            path.insert(0, start.uuid)
            return path
        explored.add(node.state.uuid)
        successors = node.getSuccessors(points,start,finish,params)
        for child in successors:
            if child.state.uuid not in explored and frontier.contains(child.state.uuid)==False:
                frontier.push(child, child.cost)
            elif frontier.contains(child.state.uuid) and frontier.pathCost(child.state.uuid) > child.cost:
                frontier.push(child,child.cost)     
        
#cost functions

def oldAngleCost(a, b, c):
    '''angle cost of going to c given we came from ab'''
    abDir = np.array(b) - np.array(a)
    bcDir = np.array(c) - np.array(b)
    difference = cluster_util.findAngle(abDir, bcDir)
    if np.isnan(difference): return 0
    else: return np.abs(difference)
    
def angleCost(a, b, c, d):
    '''prefers straighter lines'''
    abDir = np.array(b) - np.array(a)
    cdDir = np.array(d) - np.array(c)
    difference = cluster_util.findAngle(abDir, cdDir)
    if np.isnan(difference): return 0
    else: return np.abs(difference)
    
def distVarCost(a, b, c):
    #np.seterr(all='warn')
    '''prefers lines with less variance in their spacing'''
    abDist = cluster_util.findDistance(a, b)
    bcDist = cluster_util.findDistance(b, c)
    if bcDist==0:
        #shouldn't ever occur, but prevents undefined data while debugging
        return 0
    return np.abs(np.log2((1/abDist)*bcDist))

def distCost(current,step,start,goal):
    '''prefers dense lines to sparse ones'''
    stepdist = cluster_util.findDistance(current, step)
    totaldist= cluster_util.findDistance(start, goal)
    return stepdist**2/totaldist**2

def bundleSearch(scene, groups, intersection = 0,beamwidth=10):
    bgroups = groups
    global allow_intersection 
    allow_intersection = intersection
    for i in scene:
        bgroups.append(i)
    
    
#    print "number of groups:",len(groups)
    expanded = 0
    singletonCost = 1


    node = BNode(frozenset(), -1, [], 0)
    frontier = BundlePQ()
    frontier.push(node, 0)
    explored = set()
    while frontier.isEmpty() == False:
        node = frontier.pop()
        expanded += 1
        if node.getState() >= frozenset(scene):
            path = node.traceback()
            return path
        explored.add(node.state)
        successors = node.getSuccessors(scene,bgroups)
        successors.sort(key= lambda s: s.gainratio,reverse=True)
        successors = successors[0:beamwidth]
        for child in successors:

            if child.state not in explored and frontier.contains(child.state)==False:
                frontier.push(child, child.cost)

            elif frontier.contains(child.state) and frontier.pathCost(child.state) > child.cost:
                print "cheaper"
                frontier.push(child,child.cost)

    
class Node:
    def __init__(self, state, parent, action, cost,qCost):
        self.state = state
        self.parent = parent
        self.action = action
        self.icost = cost
        self.iqcost = qCost
        if parent != -1:
            self.cost = parent.cost + cost
            self.qCost = parent.qCost + qCost
        else:
            self.cost=cost
            self.qCost = qCost
            
    def getState(self):
        return self.state
    
    def getSuccessors(self, points,start,finish,params):
#        print points
#        print len(points)
        
        out = []
        if self.parent == -1: 
            for p in points:
                if self.state.uuid != p.uuid and finish.uuid!=p.uuid: 
                    aCost = angleCost(self.state.representation.middle, finish.representation.middle, self.state.representation.middle, p.representation.middle)
                    dCost = distCost(self.state.representation.middle,p.representation.middle,start.representation.middle,finish.representation.middle)
                    if aCost <= params.angle_limit and dCost < 1: # prevents it from choosing points that overshoot the target.
                        normA = params.anglevar_weight*(aCost/params.angle_limit)
                        distanceCost = dCost
                        qualityCost = normA/params.anglevar_weight
                        out.append(Node(p,self,p.uuid, distanceCost,qualityCost))
        else:
            out = []
            for p in points:
                if self.state.uuid != p.uuid: 
                    vCost = distVarCost(self.parent.state.representation.middle, self.state.representation.middle, p.representation.middle)
#                    print self.parent.state.representation.middle,self.state.representation.middle,p.representation.middle,"--",vCost/params.chain_distance_limit
                    
                    aCost = oldAngleCost(self.parent.state.representation.middle,self.state.representation.middle,p.representation.middle)
                    dCost = distCost(self.state.representation.middle,p.representation.middle,start.representation.middle,finish.representation.middle)
#                    print "dcost",dCost
                    if aCost <= params.angle_limit and dCost <= 1 and vCost/params.chain_distance_limit <= 1:
                        normV = params.distvar_weight*(vCost/params.chain_distance_limit)
                        normA = params.anglevar_weight*(aCost/params.angle_limit)
                        qualityCost = (normA+normV)/(params.distvar_weight+params.anglevar_weight)
                        out.append(Node(p,self,p.uuid,dCost,qualityCost))
        
        return out

    def traceback(self):
        solution = []
        node = self
        while node.parent != -1:
            solution.append(node.action)

            node = node.parent
        cardinality = len(solution)-1 #exclude the first node, which has cost 0
        cost = self.qCost#/cardinality
        solution.reverse()
        solution.append(cost)
        return solution

class BNode:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        if parent != -1:
            self.cost = parent.cost + cost
        else:
            self.cost=cost
        self.gain = len(self.state)-self.cost

        if len(self.state)>0:
            self.gainratio = self.gain/len(self.state)
        else: self.gainratio = 0

            
    def getState(self):
        return self.state
    
    def getSuccessors(self, points,groups):
        successors = []

        for g in groups:
            memtup = g.members

            if len(self.state.intersection(memtup))<=allow_intersection:
                asd=BNode(self.state.union(memtup),self,g.uuid,g.cost)
                successors.append(asd)
#                if asd.gain > 0:
#                    successors.append(asd)

        return successors
        

    def traceback(self):
        solution = []
        node = self
        while node.parent != -1:
            solution.append(node.action)
            node = node.parent
        cardinality = len(solution)-1 #exclude the first node, which has cost 0
        cost = self.cost
        solution.reverse()


        return solution
    
class PriorityQueue:
    '''stolen from ista 450 hw ;)'''

    def  __init__(self):  
        self.heap = []
        self.dict = dict()
    
    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)
        self.dict[item.state.uuid]=priority
    
    def contains(self,item):
        return self.dict.has_key(item)
    
    def pathCost(self,item):
        return self.dict.get(item)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item
        
    def isEmpty(self):
        return len(self.heap) == 0
    
class BundlePQ:

    def  __init__(self):  
        self.heap = []
        self.dict = dict()
    
    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)
        self.dict[item.state]=priority
    
    def contains(self,item):
        return self.dict.has_key(item)
    
    def pathCost(self,item):
        return self.dict.get(item)

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return item
        
    def isEmpty(self):
        return len(self.heap) == 0


