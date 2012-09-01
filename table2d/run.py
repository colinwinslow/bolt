#!/usr/bin/env python
from speaker import Speaker
from planar import Vec2, BoundingBox
from landmark import (GroupLineRepresentation,
                      PointRepresentation,
                      RectangleRepresentation,
                      Circle,
                      CircleRepresentation,
                      SurfaceRepresentation,
                      Scene,
                      Landmark,
                      ObjectClass,
                      Color)
from random import random
import pickle
import adapter

def construct_training_scene():
    speaker = Speaker(Vec2(0,0))
    scene = Scene(3)

    table = Landmark('table',
                     RectangleRepresentation(rect=BoundingBox([Vec2(-0.4,0.4), Vec2(0.4,1.0)])),
                     None,
                     ObjectClass.TABLE)

    obj1 = Landmark('green_cup',
                    RectangleRepresentation(rect=BoundingBox([Vec2(0.05-0.035,0.9-0.035), Vec2(0.05+0.035,0.9+0.035)]), landmarks_to_get=[]),
                    None,
                    ObjectClass.CUP,
                    Color.GREEN)

    obj2 = Landmark('blue_cup',
                    RectangleRepresentation(rect=BoundingBox([Vec2(0.05-0.035,0.7-0.035), Vec2(0.05+0.035,0.7+0.035)]), landmarks_to_get=[]),
                    None,
                    ObjectClass.CUP,
                    Color.BLUE)

    obj3 = Landmark('pink_cup',
                    RectangleRepresentation(rect=BoundingBox([Vec2(0-0.035,0.55-0.035), Vec2(0+0.035,0.55+0.035)]), landmarks_to_get=[]),
                    None,
                    ObjectClass.CUP,
                    Color.PINK)

    obj4 = Landmark('purple_prism',
                    RectangleRepresentation(rect=BoundingBox([Vec2(-0.3-0.03,0.7-0.05), Vec2(-0.3+0.03,0.7+0.05)]), landmarks_to_get=[]),
                    None,
                    ObjectClass.PRISM,
                    Color.PURPLE)

    obj5 = Landmark('orange_prism',
                    RectangleRepresentation(rect=BoundingBox([Vec2(0.3-0.03,0.7-0.05), Vec2(0.3+0.03,0.7+0.05)]), landmarks_to_get=[]),
                    None,
                    ObjectClass.PRISM,
                    Color.ORANGE)

    scene.add_landmark(table)

    for obj in (obj1, obj2, obj3, obj4, obj5):
        obj.representation.alt_representations = []
        scene.add_landmark(obj)

    return scene, speaker

if __name__ == '__main__':
    scene, speaker = construct_training_scene()
    
    lmks = [lmk for lmk in scene.landmarks.values() if not lmk.name == 'table']
    groups = adapter.adapt(lmks)
    
    for i,g in enumerate(groups):
        try: scene.add_landmark(Landmark('ol%d'%i, g, None, Landmark.LINE))
        except: print "this error is happening because of singletonbundles mixed in with the lines. Will fix soon!"
    #perspectives = [ Vec2(5.5,4.5), Vec2(6.5,6.0)]
    #speaker.talk_to_baby(scene, perspectives, how_many_each=10)

    dozen = 1
    couple = 1
    while True:
        for i in range(couple * dozen):
            location = Landmark( 'point', PointRepresentation(Vec2(random()*0.8-0.4,random()*0.6+0.4)), None, Landmark.POINT)
            trajector = location#obj2
            speaker.describe(trajector, scene, True, 1)
        # speaker.get_all_meaning_descriptions(trajector, scene, 1)
    # location = Vec2(5.68, 5.59)##Vec2(5.3, 5.5)
    # speaker.demo(location, scene)
    # all_desc = speaker.get_all_descriptions(location, scene, 1)


    # for i in range(couple * dozen):
    #     speaker.communicate(scene, False)

    # for desc in all_desc:
    #     print desc

    # r = RectangleRepresentation(['table'])
    # lmk = r.landmarks['l_edge']
    # print lmk.get_description()
    # print lmk.representation.landmarks['end'].get_description()
    # print r.landmarks['ul_corner'].get_description()

    # print r.landmarks['ul_corner'].distance_to( Vec2(0,0) )

    # representations = [r]
    # representations.extend(r.get_alt_representations())

    # location = Vec2(0,0)
    # landmarks_distances = []
    # for representation in representations:
    #     for lmk in representation.get_landmarks():
    #         landmarks_distances.append([lmk, lmk.distance_to(location)])

    # print 'Distance from POI to LLCorner landmark is %f' % r.landmarks['ll_corner'].distance_to(poi)
    # print 'Distance from POI to URCorner landmark is %f' % r.landmarks['ur_corner'].distance_to(poi)
    # print 'Distance from POI to LRCorner landmark is %f' % r.landmarks['lr_corner'].distance_to(poi)
    # print 'Distance from POI to ULCorner landmark is %f' % r.landmarks['ul_corner'].distance_to(poi)
    # print 'Distance from POI to Center landmark is %f' % r.landmarks['center'].distance_to(poi)
    # print 'Distance from POI to LEdge landmark is %f' % r.landmarks['l_edge'].distance_to(poi)
    # print 'Distance from POI to REdge landmark is %f' % r.landmarks['r_edge'].distance_to(poi)
    # print 'Distance from POI to NEdge landmark is %f' % r.landmarks['n_edge'].distance_to(poi)
    # print 'Distance from POI to FEdge landmark is %f' % r.landmarks['f_edge'].distance_to(poi)
