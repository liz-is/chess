#!/usr/bin/env python

from os import path
import logging
import numpy as np
from scipy import ndimage as ndi
from skimage import restoration
from skimage.morphology import square, closing
import skimage.filters as filters
from skimage.measure import label, regionprops
from numpy import inf
from scipy.ndimage import zoom
import warnings
from future.utils import string_types
from .helpers import (
    load_regions, load_sparse_matrix, sub_matrix_from_edges_dict, GenomicRegion
    )
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_sparse_matrix(regions, sparse_matrix):
    '''Get all the sparse HiC matrix from the bed file'''
    ix_converter = None
    if isinstance(regions, string_types):
        regions, ix_converter, _ = load_regions(regions)

    if isinstance(sparse_matrix, string_types):
        sparse_matrix = load_sparse_matrix(
            sparse_matrix, ix_converter=ix_converter)

    return regions, sparse_matrix


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
    elif zoom_factor > 1:
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = img
    return out


def get_info_feature(labels, submatrix, outfile, position, region, area, reg, bin_size):
    for feature in regionprops(labels):
        if feature.area > area:
            minr, minc, maxr, maxc = feature.bbox
            # calculate genomic coordinates
            row_coords = (region.start + (bin_size * minr), region.start + (bin_size * maxr))
            col_coords = (region.start + (bin_size * minc), region.start + (bin_size * maxc))
            chrom = region.chromosome
            row_region = f"{chrom}:{row_coords[0]}-{row_coords[1]}"
            col_region = f"{chrom}:{col_coords[0]}-{col_coords[1]}"
            # sort genomic coordinates to help with removing duplicates later
            if row_coords[0] < col_coords[0]:
                row_region, col_region = row_region, col_region
            else:
                row_region, col_region = col_region, row_region

            submat = submatrix[minr:maxr, minc:maxc]
            flat_mat = list(submat.flatten())
            flat_mat.insert(0, reg)
            flat_mat.insert(1, position)
            flat_mat.insert(2, minc)
            flat_mat.insert(3, maxc)
            flat_mat.insert(4, minr)
            flat_mat.insert(5, maxr)
            flat_mat.insert(6, row_region)
            flat_mat.insert(7, col_region)
            outfile.write(','.join(map(str, flat_mat)) + '\n')
            position += 1
    return position


def extract_structures(
    reference_edges,
    reference_regions,
    query_edges,
    query_regions,
    pairs,
    output,
    windowsize,
    sigma_spatial,
    size_medianfilter,
    closing_square,
    min_area=5
):
    pos_query = 0
    pos_reference = 0

    gained_features_file = open(path.join(output, 'gained_features.csv'), '+w')
    lost_features_file = open(path.join(output, 'lost_features.csv'), '+w')

    for pair_ix, reference_region, query_region in pairs:
        reference, ref_rs = sub_matrix_from_edges_dict(
            reference_edges,
            reference_regions,
            reference_region,
            default_weight=0.)
        query, qry_rs = sub_matrix_from_edges_dict(
            query_edges,
            query_regions,
            query_region,
            default_weight=0.)

        # it could be helpful to make this a user-defined parameter
        # maybe as a % of the input matrix area?
        # would need to be a very low % though, even 1% of area is much higher than the current value
        area = int((min_area * np.shape(query)[0]) / 100)

        # it's not clear to me why matrix dimensions are used to decide how many bins are used for histograms for
        # denoising / smoothing.
        # Would default values also be fine?
        size = np.shape(query)[0]

        # calculate log2 query/reference
        # does this only work for contact probabilities at the moment?
        # i.e. Juicer input will give unexpected results with unequal seq depth
        # I think this only works for query and ref regions that are the same size
        # is there a check for this?
        or_matrix = np.log(np.divide(query, reference))
        where_are_NaNs = np.isnan(or_matrix)
        or_matrix[where_are_NaNs] = 0.
        or_matrix[or_matrix == -inf] = 0.
        or_matrix[or_matrix == inf] = 0.
        std = np.std(or_matrix)

        positive = np.where(or_matrix > (0.5 * std), or_matrix, 0.)
        negative = np.abs(np.where(or_matrix < -(0.5 * std), or_matrix, 0.))

        # denoise
        denoise_positive = restoration.denoise_bilateral(
            positive,
            sigma_color=np.mean(positive),
            win_size=windowsize,
            sigma_spatial=sigma_spatial,
            bins=size,
            multichannel=False)
        denoise_negative = restoration.denoise_bilateral(
            negative,
            sigma_color=np.mean(negative),
            win_size=windowsize,
            sigma_spatial=sigma_spatial,
            bins=size,
            multichannel=False)
        # smooth
        filter_positive = ndi.median_filter(
            denoise_positive, size_medianfilter)
        filter_negative = ndi.median_filter(
            denoise_negative, size_medianfilter)

        # binarise
        if np.all(filter_positive == 0.):
            # I think the aim of this is to avoid attempting thresholding if all values are the same
            # so should be `filter_positive`?
            threshold_pos = positive
        else:
            filter1 = filters.threshold_otsu(filter_positive, nbins=size)
            threshold_pos = filter_positive > filter1

        if np.all(filter_negative == 0.):
            # should be `filter_negative`?
            threshold_neg = negative
        else:
            filter2 = filters.threshold_otsu(filter_negative, nbins=size)
            threshold_neg = filter_negative > filter2

        # Close morphology
        img1 = closing(threshold_pos, square(closing_square))
        label_x1 = label(img1)
        img2 = closing(threshold_neg, square(closing_square))
        label_x2 = label(img2)

        # get Hi-C bin size / resolution in bp
        # needed for calculating genomic coordinates from pixel coordinates
        # is there an easier / more robust way of doing this?
        hic_bin_size = reference_regions[0].end - reference_regions[0].start + 1

        # get output (file with label and submatrices)
        pos_query = get_info_feature(
            label_x1, query, gained_features_file, pos_query, query_region, area, pair_ix, hic_bin_size)
        pos_reference = get_info_feature(
            label_x2, reference, lost_features_file, pos_reference, reference_region, area, pair_ix, hic_bin_size)
