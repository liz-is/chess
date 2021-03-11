#!/usr/bin/env python

import os
import logging
import numpy as np
from scipy import ndimage as ndi
from skimage import restoration
from skimage.morphology import square, closing
import skimage.filters as filters
from skimage.measure import label, regionprops
from numpy import inf
import warnings
from .helpers import sub_matrix_from_edges_dict
import pandas as pd
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_info_feature_square(labels, submatrix, fc_matrix, outfile, position, region, min_feature_size, reg, bin_size):
    feature_data_list = []
    for feature in regionprops(labels):
        minr, minc, maxr, maxc = feature.bbox
        # calculate bbox edge size in pixels
        # if shortest side is less than min_feature_size, skip this region
        if min(maxc - minc, maxr - minr) >= min_feature_size:
            # calculate genomic coordinates
            row_coords = (region.start + (bin_size * minr), region.start + (bin_size * maxr))
            col_coords = (region.start + (bin_size * minc), region.start + (bin_size * maxc))
            chrom = region.chromosome
            row_region = f"{chrom}:{row_coords[0]}-{row_coords[1]}"
            col_region = f"{chrom}:{col_coords[0]}-{col_coords[1]}"
            # sort genomic coordinates to help with removing duplicates later
            if row_coords[0] > col_coords[0]:
                row_region, col_region = col_region, row_region

            submat = submatrix[minr:maxr, minc:maxc]
            mean_fc = np.mean(fc_matrix[minr:maxr, minc:maxc])

            flat_mat = ";".join(str(x) for x in submat.flatten())

            output_data = {
                "region_pair_id": reg,
                "feature_idx": position,
                "minc": minc,
                "maxc": maxc,
                "minr": minr,
                "maxr": maxr,
                "row_coords": row_region,
                "column_coords": col_region,
                "mean_query_ref_foldchange": mean_fc,
                "matrix_data": flat_mat
            }
            feature_data_list.append(output_data)
            position += 1

    df = pd.DataFrame(feature_data_list, columns = ["region_pair_id", "feature_idx", "minc", "maxc", "minr", "maxr",
                                                    "row_coords", "column_coords", "mean_query_ref_foldchange",
                                                    "matrix_data"])
    # probably not needed any more because lower triangle is masked
    df = df[-df.duplicated(subset=['row_coords', 'column_coords'])]

    df.to_csv(outfile, mode='a', index=False, header=not os.path.exists(outfile))
    return position


def extract_structures_square(
    reference_edges,
    reference_regions,
    query_edges,
    query_regions,
    hic_bin_size,
    pairs,
    output,
    windowsize,
    sigma_spatial,
    size_medianfilter,
    closing_square,
    min_feature_size,
    plot=None
):
    pos_query = 0
    pos_reference = 0

    gained_features_file = os.path.join(output, 'gained_features.csv')
    if os.path.exists(gained_features_file):
        logger.warning(f'Output file {gained_features_file} already exists and will be overwritten!')
        os.remove(gained_features_file)
    lost_features_file = os.path.join(output, 'lost_features.csv')
    if os.path.exists(lost_features_file):
        logger.warning(f'Output file {lost_features_file} already exists and will be overwritten!')
        os.remove(lost_features_file)

    plot_file = None
    if plot is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        plot_file = PdfPages(plot)

    try:
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
                multichannel=False)
            denoise_negative = restoration.denoise_bilateral(
                negative,
                sigma_color=np.mean(negative),
                win_size=windowsize,
                sigma_spatial=sigma_spatial,
                multichannel=False)
            # smooth
            filter_positive = ndi.median_filter(
                denoise_positive, size_medianfilter)
            filter_negative = ndi.median_filter(
                denoise_negative, size_medianfilter)

            # to mask lower triangle
            lower_tri_idx = np.tril_indices(filter_positive.shape[0], 1)

            # binarise
            if np.all(filter_positive == 0.):
                threshold_pos = filter_positive
            else:
                filter_positive[lower_tri_idx] = 0.
                filter1 = filters.threshold_otsu(filter_positive)
                threshold_pos = filter_positive > filter1

            if np.all(filter_negative == 0.):
                threshold_neg = filter_negative
            else:
                filter_positive[lower_tri_idx] = 0.
                filter2 = filters.threshold_otsu(filter_negative)
                threshold_neg = filter_negative > filter2

            # Close morphology
            img1 = closing(threshold_pos, square(closing_square))
            label_x1 = label(img1)
            img2 = closing(threshold_neg, square(closing_square))
            label_x2 = label(img2)

            # get output (file with label and submatrices)
            pos_query = get_info_feature_square(
                label_x1, query, or_matrix, gained_features_file, pos_query,
                query_region, min_feature_size, pair_ix, hic_bin_size)
            pos_reference = get_info_feature_square(
                label_x2, reference, or_matrix, lost_features_file, pos_reference,
                reference_region, min_feature_size, pair_ix, hic_bin_size)

            if plot_file is not None:
                plot_features(plot_file, reference, query, or_matrix, label_x1, label_x2, min_feature_size,
                              cmap='germany', reference_region=str(reference_region), query_region=str(query_region))

    finally:
        if plot_file is not None:
            plot_file.close()


def plot_features(plot_file, reference, query, or_matrix, label_x1, label_x2, min_feature_size,
                  cmap, linecolor='royalblue', vmin=1e-3, vmax=1e-1,
                  reference_region='', query_region=''):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm
    import fanc.plotting
    # for colormap only, need to replace this!

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 5), ncols=3)
    ax1.imshow(reference, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)

    # higher in ref
    for feature in regionprops(label_x2):
        # take regions with large enough areas
        minr, minc, maxr, maxc = feature.bbox
        # calculate bbox edge size in pixels
        # if shortest side is less than min_feature_size, skip this region
        if min(maxc - minc, maxr - minr) >= min_feature_size:
            # draw rectangle around segmented coins
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=linecolor, linewidth=2)
            ax1.add_patch(rect)
    ax1.set_axis_off()

    ax2.imshow(query, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)

    # higher in query
    for feature in regionprops(label_x1):
        # take regions with large enough areas
        minr, minc, maxr, maxc = feature.bbox
        # calculate bbox edge size in pixels
        # if shortest side is less than min_feature_size, skip this region
        if min(maxc - minc, maxr - minr) >= min_feature_size:
            # draw rectangle around segmented coins
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=linecolor, linewidth=2)
            ax2.add_patch(rect)
    ax2.set_axis_off()

    ax3.imshow(or_matrix, cmap="RdBu_r", vmax=1, vmin=-1)
    ax3.set_axis_off()

    ax1.set_title('reference' + ' ' + reference_region)
    ax2.set_title('query' + ' ' + query_region)
    ax3.set_title('query / reference')
    plt.tight_layout()
    plot_file.savefig()