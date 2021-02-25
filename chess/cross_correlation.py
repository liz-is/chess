#!/usr/bin/env python

import matplotlib
matplotlib.use('pdf')

import logging
from os import path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.signal.signaltools import correlate2d as c2d
import pandas as pd
import itertools
from tqdm import tqdm
from kneed import KneeLocator

"""Compute 2D cross-correlation and get clusters of structures"""


def correlate2d(file, output_folder, pairs):
    logger = logging.getLogger('')
    # load region pairs
    pairs_dict = defaultdict(str)
    for pair_ix, reference_region, query_region in pairs:
        pairs_dict[pair_ix] = str(reference_region)+'-'+str(query_region)

    # load features
    df = pd.read_csv(file, nrows=1)
    if len(df.columns) > 9:
        all_arrays = defaultdict(list)
        information_regions = defaultdict(list)
        with open(file, 'r') as r:
            for line in r:
                region, position, x_min, x_max, y_min, y_max = line.split(',')[:6]
                pair_id = pairs_dict[int(region)]
                line_float = [float(x) for x in line.split(',')[6:]]
                height, width = int(y_max) - int(y_min), int(x_max) - int(x_min)
                mat = np.asanyarray(line_float).reshape(int(height), int(width))
                all_arrays[int(position)].append(mat)
                information_regions[int(position)].append((pair_id, int(position)))

    else:
        df = pd.read_csv(file).set_index('feature_idx')
        df['height'] = df['maxc'] - df['minc']
        df['width'] = df['maxr'] - df['minr']
        all_arrays = {feature_id: np.asfarray(matrix_data.split(';')).reshape(height, width) for
                      (feature_id, matrix_data, height, width) in
                      zip(df.index, df['matrix_data'], df['height'], df['width'])}
        information_regions = {feature_id: (pair_id, feature_id) for (feature_id, pair_id) in
                               zip(df.index, df['region_pair_id'])}

    logger.info(
        '[MAIN]: All submatrices loaded, starting 2D cross-correlation')

    tag = file.split('/')[-1].split('_')[0]
    # calculate correlations
    correlation_dataframe = pd.DataFrame(
        index=df.index,
        columns=df.index
    )
    for a, b in tqdm(itertools.combinations(df.index, 2)):
        c11 = c2d(all_arrays[a], all_arrays[b], mode='same')
        c112 = c2d(all_arrays[b], all_arrays[a], mode='same')
        transp1 = c2d(np.fliplr(all_arrays[a]), all_arrays[b], mode='same')
        transp2 = c2d(np.fliplr(all_arrays[b]), all_arrays[a], mode='same')
        best = max(c11.max(), c112.max(), transp1.max(), transp2.max())
        correlation_dataframe.loc[a, b] = best
        correlation_dataframe.loc[b, a] = best
    logger.info('[MAIN]: 2D cross-correlation done')

    correlation_dataframe = correlation_dataframe.replace(np.Inf, np.nan)
    correlation_dataframe = correlation_dataframe.replace(-np.Inf, np.nan)
    scaled = scale(correlation_dataframe.fillna(0.))

    # save dataframe ##
    correlation_dataframe.to_csv(
        path.join(output_folder, 'correlation_dataframe_%s.csv' % (tag)))

    sum_of_squared_distances = []
    K = range(1, min(scaled.shape[0], 15))
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(scaled)
        sum_of_squared_distances.append(km.inertia_)

    kn = KneeLocator(
        range(1, len(sum_of_squared_distances)+1),
        sum_of_squared_distances,
        curve='convex',
        direction='decreasing')
    optimal_number_clusters = kn.knee

    plt.figure(figsize=(4, 3))
    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(
        range(1, len(sum_of_squared_distances)+1),
        sum_of_squared_distances,
        'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig(path.join(output_folder, 'Elbow_index_%s.pdf' % tag))

    # save positions and belonging cluster
    logger.info('[MAIN]: Classification of the features')
    kmeans = KMeans(
        n_clusters=optimal_number_clusters,
        random_state=0,
        precompute_distances=True).fit(scaled)
    w = open(path.join(
        output_folder,
        'subregions_%s_clusters_%s.tsv' % (str(optimal_number_clusters), tag)),
        'a+')
    for n, i in enumerate(list(kmeans.labels_)):
        t1, t2 = information_regions[n][0]
        w.write('{}\t{}\t{}\n'.format('Cluster ' + str(i), t1, t2))
    w.close()
