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
import warnings
from .helpers import sub_matrix_from_edges_dict
import pandas as pd
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_info_feature_square(labels, submatrix, outfile, position, region, area, reg, bin_size):
    feature_data_list = []
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
                "matrix_data": flat_mat
            }
            feature_data_list.append(output_data)
            position += 1

    df = pd.DataFrame(feature_data_list)
    df = df[-df.duplicated(subset=['row_coords', 'column_coords'])]

    df.to_csv(outfile, mode='a', index=False, header=not path.exists(outfile))
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
    min_area=5,
    plot=None
):
    pos_query = 0
    pos_reference = 0

    gained_features_file = path.join(output, 'gained_features.csv')
    lost_features_file = path.join(output, 'lost_features.csv')

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

            # it could be helpful to make this a user-defined parameter
            # maybe as a % of the input matrix area?
            # would need to be a very low % though, even 1% of area is much higher than the current value
            area = int((min_area * np.shape(query)[0]) / 100)

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

            # binarise
            if np.all(filter_positive == 0.):
                threshold_pos = filter_positive
            else:
                filter1 = filters.threshold_otsu(filter_positive)
                threshold_pos = filter_positive > filter1

            if np.all(filter_negative == 0.):
                threshold_neg = filter_negative
            else:
                filter2 = filters.threshold_otsu(filter_negative)
                threshold_neg = filter_negative > filter2

            # Close morphology
            img1 = closing(threshold_pos, square(closing_square))
            label_x1 = label(img1)
            img2 = closing(threshold_neg, square(closing_square))
            label_x2 = label(img2)

            # get output (file with label and submatrices)
            pos_query = get_info_feature_square(
                label_x1, query, gained_features_file, pos_query, query_region, area, pair_ix, hic_bin_size)
            pos_reference = get_info_feature_square(
                label_x2, reference, lost_features_file, pos_reference, reference_region, area, pair_ix, hic_bin_size)

            if plot_file is not None:
                plot_features(plot_file, reference, query, or_matrix, label_x1, label_x2, area, cmap='germany',
                              reference_region=str(reference_region), query_region=str(query_region))

    finally:
        if plot_file is not None:
            plot_file.close()


def plot_features(plot_file, reference, query, or_matrix, label_x1, label_x2, area,
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
        if feature.area >= area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = feature.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=linecolor, linewidth=2)
            ax1.add_patch(rect)
    ax1.set_axis_off()

    ax2.imshow(query, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)

    # higher in query
    for feature in regionprops(label_x1):
        # take regions with large enough areas
        if feature.area >= area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = feature.bbox
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