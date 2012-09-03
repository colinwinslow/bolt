#!/usr/bin/env python
from __future__ import division

from random import random

from sentence_from_location import (
    generate_sentence,
    accept_correction,
    Point
)

from table2d.run import construct_training_scene
from table2d.landmark import Landmark, PointRepresentation
from nltk.metrics.distance import edit_distance
from planar import Vec2
from utils import logger, m2s
import numpy as np
from matplotlib import pyplot as plt

from location_from_sentence import get_all_sentence_posteriors

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, default=1)
    parser.add_argument('-l', '--location', type=Point)
    parser.add_argument('--consistent', action='store_true')
    args = parser.parse_args()

    scene, speaker = construct_training_scene()
    table = scene.landmarks['table'].representation.get_geometry()

    window = 10
    scales = [100]
    min_dists = []
    max_dists = []
    avg_min = []
    max_mins = []

    step = 0.02
    all_heatmaps_dict, xs, ys = speaker.generate_all_heatmaps(scene, step=step)
    x = np.array( [list(xs-step*0.5)]*len(ys) )
    y = np.array( [list(ys-step*0.5)]*len(xs) ).T

    all_heatmaps_tuples = []
    for lmk, d in all_heatmaps_dict.items():
        for rel, heatmap in d.items():
            all_heatmaps_tuples.append( (lmk,rel,heatmap) )
    lmks, rels, heatmaps = zip(*all_heatmaps_tuples)
    meanings = zip(lmks,rels)

    demo_sentence = 'near to the left edge of the table'

    for iteration in range(args.num_iterations):

        posteriors = np.array(get_all_sentence_posteriors(demo_sentence, meanings))
        print sorted(zip(posteriors, meanings))
        posteriors /= posteriors.sum()
        print sorted(zip(posteriors, meanings))
        big_heatmap = None
        for p,h in zip(posteriors, heatmaps):
            if big_heatmap is None:
                big_heatmap = p*h
            else:
                big_heatmap += p*h

        print big_heatmap.shape
        print xs.shape, ys.shape

        probabilities = big_heatmap.reshape( (len(xs),len(ys)) ).T

        # print probabilities

        #print self.get_entropy(probabilities)
        plt.pcolor(x, y, probabilities, cmap = 'jet', edgecolors='none', alpha=0.7)
        plt.colorbar()
        plt.show()
        exit()




        logger('Iteration %d' % iteration)
        scale = 1000
        rand_p = Vec2(random()*table.width+table.min_point.x, random()*table.height+table.min_point.y)
        meaning, sentence = generate_sentence(rand_p, args.consistent, scene, speaker)

        logger( 'Generated sentence: %s' % sentence)

        trajector = Landmark( 'point', PointRepresentation(rand_p), None, Landmark.POINT )
        landmark, relation = meaning.args[0], meaning.args[3]
        head_on = speaker.get_head_on_viewpoint(landmark)
        all_descs = speaker.get_all_meaning_descriptions(trajector, scene, landmark, relation, head_on, 1)

        distances = []
        for desc in all_descs:
            distances.append([edit_distance( sentence, desc ), desc])

        distances.sort()
        print distances

        min_dists.append(distances[0][0])
        avg_min.append( np.mean(min_dists[-window:]) )
        max_mins.append( max(min_dists[-window:]) )

        correction = distances[0][1]
        accept_correction( meaning, correction, update_scale=scale )

        print np.mean(min_dists), avg_min, max_mins
        print;print

    plt.plot(avg_min, 'bo-')
    plt.plot(max_mins, 'rx-')
    plt.show()
