from planar import Vec2, BoundingBox
import numpy as np
import SceneEval
import landmark


objects = []
def adapt(landmarks):
    '''takes a scene object and returns a list of lists of objects that form groups'''
    
    
    #working on eliminating the adapter
    
    bundles = SceneEval.sceneEval(landmarks)

    landmarkDict = dict()
    for l in landmarks:
        landmarkDict[l.uuid]=l
    

    return  bundles
