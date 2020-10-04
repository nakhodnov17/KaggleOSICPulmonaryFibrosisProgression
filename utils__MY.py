import os
import sys
import platform
from collections import defaultdict

import tabulate
import numpy as np
import pandas as pd
import scipy
import sparse
import scipy.sparse

import cv2
import skimage
from skimage import measure, filters, segmentation
import imageio

from pydicom import dcmread, multival

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from IPython.display import clear_output, display


# Global variables
IMAGE_PATH = '../input/osic-pulmonary-fibrosis-progression/' if 'linux' in platform.platform().lower() else 'data/'
PROCESSED_PATH = 'FIX IT!' if 'linux' in platform.platform().lower() else 'data/processed-data/'  # TODO: fix this line

test_patients = sorted(os.listdir(f'{IMAGE_PATH}/test/'))
train_patients = sorted(os.listdir(f'{IMAGE_PATH}/train/'))


# Functions
def subplots_3d(nrows=1, ncols=1, figsize=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape([nrows, ncols])
    
    for idx in range(len(axes)):
        for jdx in range(len(axes[idx])):
            axes[idx][jdx].remove()
            axes[idx][jdx] = fig.add_subplot(nrows, ncols, 1 + idx * ncols + jdx,projection='3d')
            
    if nrows == 1:
        return fig, axes[0].tolist()
    return fig, axes.tolist()

def plot_3d(ax, image, stride, threshold=700, color="navy"):
    if isinstance(stride, (int, float)):
        stride = (stride, stride, stride)
    
    image = image[::stride[0], ::stride[1], ::stride[2]]
        
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces,_,_ = skimage.measure.marching_cubes(p, threshold)
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.2)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
def sample_slices(images, n_slices, masks=None, alpha=0.2, cmap_name=None):
    nrows, ncols = n_slices // 4 + ((n_slices % 4) > 0), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).reshape(-1)

    idxs = np.sort(np.random.choice(np.arange(len(images)), nrows * ncols, replace=False))
    
    for ax, idx in zip(axes, idxs):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(images[idx], cmap=plt.get_cmap(cmap_name))
        if masks is not None:
            ax.imshow(masks[idx], alpha=alpha)
#             ax.imshow(masks[idx], alpha=alpha, cmap=plt.get_cmap(cmap_name))
        ax.grid(False)
        ax.set_title(f'Slice {idx}')
    plt.show()

def getkey(dcm, key):
    try:
        base_value = dcm[key]
    except:
        print(dcm.PatientID, dcm.InstanceNumber, key)
        return 'KeyError'
    try:
        return base_value.value
    except:
        return base_value

def transform_to_hu(ct):
    slope = ct.RescaleSlope
    intercept = ct.RescaleIntercept
    image = ct.pixel_array.astype(np.int16)
    if getkey(ct, 'PatientID') in {'ID00128637202219474716089', 'ID00026637202179561894768'}:
        intercept += 1000
    if getkey(ct, 'PatientID') in {'ID00132637202222178761324'}:
        intercept += 4096 / 4 * 3 - 1200
    
    # some images has ousige pixel-values corresponging water
    # fix those images
    zero_cols = np.argwhere(np.sum(ct.pixel_array, axis=0) == 0).reshape(-1)
    zero_rows = np.argwhere(np.sum(ct.pixel_array, axis=1) == 0).reshape(-1)
    image[zero_rows, :] = -1000 - np.int16(intercept)
    image[:, zero_cols] = -1000 - np.int16(intercept)
    
    # convert to HU
    if slope != 1:
        image = (slope * image.astype(np.float64)).astype(np.int16)
    image += np.int16(intercept)
    
    # convert ouside pixel-values to air:
    # I'm using <= -1000 to be sure that other defaults are captured as well
    image[image <= -1000] = -1000
    
    return ct, image

def crop(image, center, size):
    return image.copy()[
        center[0] - size[0] // 2:center[0] + size[0] // 2
    ][:, center[1] - size[1] // 2:center[1] + size[1] // 2]

transformations_in_train = defaultdict(lambda: lambda x: x, {
    'ID00014637202177757139317': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512)),
    'ID00067637202189903532242': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512)),
    'ID00086637202203494931510': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512)),
    'ID00094637202205333947361': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512)),
    'ID00122637202216437668965': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512)),
    'ID00240637202264138860065': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512)),
    'ID00419637202311204720264': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512))
})

# # images that needs transformation in train set:
# '''
#     ID00115637202211874187958 1302 1302
#     ID00288637202279148973731 632 632
#     ID00358637202295388077032 632 632 
#     ID00009637202177434476278 768 768
#     ID00015637202177877247924 768 768
#     ID00025637202179541264076 768 768
#     ID00026637202179561894768 768 768
#     ID00027637202179689871102 768 768
#     ID00038637202182690843176 768 768
#     ID00042637202184406822975 768 768
#     ID00078637202199415319443 768 768
#     ID00082637202201836229724 768 768
#     ID00089637202204675567570 768 768
#     ID00105637202208831864134 768 768
#     ID00108637202209619669361 768 768
#     ID00110637202210673668310 768 768
#     ID00128637202219474716089 768 768
#     ID00129637202219868188000 768 768
#     ID00132637202222178761324 768 768
#     ID00169637202238024117706 768 768
#     ID00173637202238329754031 768 768
#     ID00183637202241995351650 768 768
#     ID00214637202257820847190 768 768
#     ID00216637202257988213445 768 768
#     ID00242637202264759739921 768 768
#     ID00248637202266698862378 768 768
#     ID00285637202278913507108 768 768
#     ID00290637202279304677843 768 768
#     ID00291637202279398396106 768 768
#     ID00309637202282195513787 768 768
#     ID00343637202287577133798 768 768
#     ID00344637202287684217717 768 768
#     ID00351637202289476567312 768 768
#     ID00367637202296290303449 768 768
#     ID00388637202301028491611 768 768
#     ID00414637202310318891556 768 768
#     ID00421637202311550012437 768 768
# '''

transformations_in_test = defaultdict(lambda: lambda x: x, {
    'ID00419637202311204720264': lambda x: crop(x, (x.shape[0] // 2, x.shape[1] // 2), (512, 512))
})
    
# # images that needs transformation in test set:
# '''
#     ID00421637202311550012437 768 768
# '''

transformations = {
    'test': transformations_in_test,
    'train': transformations_in_train
}

def int16touint8(image):
    result = image.copy().astype(np.int16)
    result = result - result.min() + 10
    result = result / (result.max() + 10)
    return (result * 255.0).astype(np.uint8)

def resample(image, target_shape, SliceThickness, PixelSpacing):
    factors = np.array(target_shape, dtype=np.float32) / np.array(image.shape, dtype=np.float32)
    image = scipy.ndimage.interpolation.zoom(image.astype(np.float32), factors, mode='nearest', order=1)

    # Determine current pixel spacing
    spacing = np.array([SliceThickness] + list(PixelSpacing), dtype=np.float32)
    new_spacing = spacing / factors

    return image, new_spacing

def segmentate_patient(mode, patient_n):
    base_path = os.path.join(IMAGE_PATH, mode)
    patient = sorted(os.listdir(base_path))[patient_n]
    patient_path = os.path.join(base_path, patient)
    
    all_images, all_lungs, all_residuals, all_masks = [], [], [], []
    meta_data = {
        'InstanceNumber' : [],
        'PixelSpacing' : [], 'SliceLocation' : [], 'SliceThickness' : [],           
        'PositionReferenceIndicator' : [], 'PatientPosition' : [], 'TableHeight' : [],         
        'WindowCenter' : [], 'WindowWidth' : []                
    }
    for idx, ct_name in enumerate(sorted(os.listdir(patient_path), key=lambda x: int(x.split('.')[0]))):
        if patient_n == 32:
            if idx > 506:
                break
        ct_path = os.path.join(patient_path, ct_name)
        ct, ct_image = transform_to_hu(dcmread(ct_path))
        ct_image = transformations[mode][patient](ct_image)
        all_images.append(ct_image)
        
        lungs, residual, mask = segment_lungs(ct_image, patient_n, idx, display=False)
        
        all_lungs.append(lungs)
        all_residuals.append(residual)        
        all_masks.append(mask)
        
        for key in meta_data.keys():
            meta_data[key].append(getkey(ct, key))
            
    return all_images, all_lungs, all_residuals, all_masks, meta_data

def merge(all_lungs, all_residuals, meta_data):
    def _issorted(_array):
        return np.all(_array[:-1] <= _array[1:]) or np.all(_array[:-1] >= _array[1:])
    
    SliceLocations = np.array(meta_data['SliceLocation'])
    InstanceNumbers = np.array(meta_data['InstanceNumber'])
    
#     assert _issorted(SliceLocations)
#     assert _issorted(InstanceNumbers)
    
    return np.stack(all_lungs), np.stack(all_residuals)

def _get_max_quantile(_array, _thresh=0.9):
    sorted_array = np.sort(_array)[::-1]
    thresh_idx =  np.argwhere((np.cumsum(sorted_array) / np.sum(_array) > _thresh)).reshape(-1)[0]
    return sorted_array[thresh_idx]

def segment_lungs(image, patient_n, image_n, display=False):
    def _contour_border_distance(_contour, shape):
        _distance = 2 * shape[0]
        _vdistance = 2 * shape[0]
        _hdistance = 2 * shape[0]
        for (point, ) in _contour:
            # image[point[1], point[0]]
            _distance = min(_distance, point[0], point[1], shape[1] - point[0], shape[0] - point[1])
            _vdistance = min(_vdistance, point[1], shape[0] - point[1])
            _hdistance = min(_hdistance, point[0], shape[1] - point[0])
        return _distance, _vdistance, _hdistance
    
    def _filter_contours(_mask):
        # Find all segments on the mask
        contours, hierarchy = cv2.findContours(_mask.astype(np.uint8), 1, 2)
        if len(contours) == 0:
            return _mask
        
        # Evaluate some statistics for each segment like area, perimeter, shape of bounding box, distance to borders
        countours_metrics = []
        for contour in contours:
            ((_, _), (horizontal_range, vertical_range), _) = cv2.minAreaRect(contour)

            countours_metrics.append(
                [
                    horizontal_range, vertical_range, 
                    cv2.arcLength(contour, True), cv2.contourArea(contour), 
                    _contour_border_distance(contour, image.shape)
                ]
            )
        
        # If a perimeter to small -- drop segment
        perimeter_thresh_1 = 2 * (2 * (_mask.shape[0] / 512.0) + 2 * (_mask.shape[1] / 512.0))
        # If a perimeter less than 10-th quantile -- drop segment
        perimeter_thresh_2 = np.quantile([_[2] for _ in countours_metrics], 0.1) if len(countours_metrics) > 7 else 0.0
        perimeter_thresh = max(perimeter_thresh_1, perimeter_thresh_2)

        # If an area to small -- drop segment
        area_thresh_1 = 6 * 6 * (_mask.shape[0] / 512.0)  * (_mask.shape[1] / 512.0)
        # If anarea less than 10-th quantile -- drop segment
        area_thresh_2 = np.quantile([_[3] for _ in countours_metrics], 0.1) if len(countours_metrics) > 7 else 0.0
        # Save the lagrest segments -- segments that overlap more than 95% of area of all segments
        area_thresh_3 = _get_max_quantile([_[3] for _ in countours_metrics], 0.95)
        area_thresh = max(area_thresh_1, area_thresh_2, area_thresh_3)
        
        # If a segment too close to the top or bottom border -- drop segment
        vdistance_to_border_thresh = 30.0 * (_mask.shape[0] / 512.0)
        # If a segment too close to the left or right border -- drop segment
        hdistance_to_border_thresh = 3.0 * (_mask.shape[1] / 512.0)
        
        for contour, (horizontal_range, vertical_range, perimeter, area, distances) in zip(contours, countours_metrics):
            distance, vdistance, hdistance = distances
            if_small_area = area < area_thresh
            if_small_high = vertical_range < 2.0
            if_small_width = horizontal_range < 2.0
            if_small_perimeter = perimeter < perimeter_thresh
            if_too_close_to_vborder = vdistance < vdistance_to_border_thresh
            if_too_close_to_hborder = hdistance < hdistance_to_border_thresh
            # If a segment too wide in compare to its height -- drop segment 
            if_too_long_1 = max(horizontal_range / vertical_range, vertical_range / horizontal_range) > 7.5
            # If a segment too wide and too flat -- drop segment 
            if_too_long_2 = horizontal_range > 0.5 * _mask.shape[0] and vertical_range < 0.15 * _mask.shape[1]
            # If a segment really too wide -- drope segment
            if_too_long_3 = horizontal_range > 0.87 * _mask.shape[0] or vertical_range > 0.87 * _mask.shape[1]
            
            if (
                if_small_area or 
                if_small_high or 
                if_small_width or 
                if_small_perimeter or
                if_too_close_to_vborder or
                if_too_close_to_hborder or
                if_too_long_1 or
                if_too_long_2 or 
                if_too_long_3
            ):        
                _mask = cv2.drawContours(
                    _mask.astype(np.uint8), [contour],
                    contourIdx=-1, color=(0), thickness=-1).astype(np.bool)
                if display:
                    print('Remove: ', (horizontal_range, vertical_range, perimeter, area, distances))
                    table = tabulate.tabulate([[
                        if_small_area,
                        if_small_high,
                        if_small_width,
                        if_small_perimeter,
                        if_too_close_to_vborder,
                        if_too_close_to_hborder,
                        if_too_long_1,
                        if_too_long_2,
                        if_too_long_3
                    ]], headers=[
                        'smll_area',
                        'smll_high',
                        'smll_width',
                        'smll_perimeter',
                        'close_to_vbrd',
                        'close_to_hbrd',
                        'long_1',
                        'long_2',
                        'long_3'
                    ])
                    print(table)
            else:
                if display:
                    print('Do not remove: ', (horizontal_range, vertical_range, perimeter, area, distances))
        
        return _mask

    thresh = -700
    if patient_n == 51:
        thresh = -950
    if patient_n == 57:
        if image_n >= 9:
            thresh = -800
        else:
            thresh = -950
    if patient_n == 63:
        thresh = -900
    if patient_n == 64:
        thresh = -900
    if patient_n == 78:
        thresh = -800
    if patient_n == 96:
        thresh = -900
    if patient_n == 108:
        thresh = -300
        
    thresh_mask = image <= thresh
    
    if patient_n == 30:
        if image_n in {0, 1, 2, 3, 4, 5, 6, 7, 8}:
            thresh_mask = np.zeros_like(thresh_mask)
    
    if patient_n == 54:
        if image_n in {0}:
            thresh_mask = np.zeros_like(thresh_mask)

    if patient_n == 57:
        if image_n in {0, 1, 2, 3}:
            thresh_mask = np.zeros_like(thresh_mask)
            
    if patient_n == 63:
        if image_n in {315, 316, 318, 319, 322, 323, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337}:
            thresh_mask = np.zeros_like(thresh_mask)
            
    if patient_n == 64:
        if image_n in {_ for _ in range(34)}:
            thresh_mask = np.zeros_like(thresh_mask)
            
    if patient_n == 78:
        if image_n in {0, 1, 2, 3, 4, 5}:
            thresh_mask = np.zeros_like(thresh_mask)
            
    if patient_n == 80:
        if image_n in {0, 1, 2, 3}:
            thresh_mask = np.zeros_like(thresh_mask)
    
    if patient_n == 86:
        if image_n in {_ for _ in range(127)}:
            thresh_mask = np.zeros_like(thresh_mask)
    
    if patient_n == 115:
        if image_n in {0, 1}:
            thresh_mask = np.zeros_like(thresh_mask)
    
    if patient_n == 121:
        if image_n in {0, 1, 2, 3, 4, 5}:
            thresh_mask = np.zeros_like(thresh_mask)
    
    if patient_n >= 26:
        # Remove small holes and disconnections (Fixes patient №26. Need to check first 26 patients)
        # Bad for some patients with № < 26
        kernel = skimage.morphology.disk(3)
        thresh_mask = skimage.morphology.binary_closing(thresh_mask, selem=kernel)
    
    thresh_mask = skimage.segmentation.clear_border(thresh_mask)
    
    if patient_n == 57:
        if image_n in {4, 5, 6}:    
            kernel = np.ones([3, 3], dtype=np.bool)
            thresh_mask = skimage.morphology.erosion(thresh_mask, selem=kernel)
            kernel = np.ones([2, 2], dtype=np.bool)
            thresh_mask = skimage.morphology.erosion(thresh_mask, selem=kernel)
    
    # Smooth image with
    lungs_mask = skimage.filters.median(thresh_mask)
    
    # Remove small holes and disconnections
    kernel = skimage.morphology.disk(7)
    lungs_mask = skimage.morphology.binary_closing(lungs_mask, selem=kernel)
    
    # Expand verticaly
    kernel = np.ones([3, 1], dtype=np.bool)
    lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    # Expand horizontaly
    kernel = np.ones([1, 7], dtype=np.bool)
    lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)

    if patient_n == 68:
        if image_n in {187}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    if patient_n == 84:
        if image_n in {67, 68}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
         
    # Remove some garbage
    lungs_mask = _filter_contours(lungs_mask)

    if patient_n == 68:
        if image_n in {187}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)    
    if patient_n == 84:
        if image_n in {67, 68}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    
    # Squeeze image horizontaly back
    kernel = np.ones([1, 7], dtype=np.bool)
    lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    # Squeeze image verticaly back
    kernel = np.ones([3, 1], dtype=np.bool)
    lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    
    # Fixes patient №55 (works well for all patients)
    kernel = np.ones([3, 1], dtype=np.bool)
    lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    
    if patient_n == 67:
        if image_n in {51, 52, 53, 54, 55, 56, 57, 58, 59, 69, 70, 83}:
            kernel = np.ones([5, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    if patient_n == 86:
        if image_n in {380, 383, 384, 385, 386, 387, 388, 389, 390}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
    if patient_n == 121:
        if image_n in {148, 149, 150, 151, 152, 153, 154, 155, 156, 157}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.erosion(lungs_mask, selem=kernel)
            
    # Remove small holes
    lungs_mask = scipy.ndimage.binary_fill_holes(lungs_mask)
    # Fixes patient №55 (works well for all patients)
    kernel = np.ones([3, 1], dtype=np.bool)
    lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    
    if patient_n == 67:
        if image_n in {51, 52, 53, 54, 55, 56, 57, 58, 59, 69, 70, 83}:
            kernel = np.ones([5, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    if patient_n == 86:
        if image_n in {380, 383, 384, 385, 386, 387, 388, 389, 390}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    if patient_n == 121:
        if image_n in {148, 149, 150, 151, 152, 153, 154, 155, 156, 157}:
            kernel = np.ones([3, 1], dtype=np.bool)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
            lungs_mask = skimage.morphology.dilation(lungs_mask, selem=kernel)
    
    lungs = lungs_mask * image
    residual = (1.0 - lungs_mask) * image
    
    lungs[lungs == 0] = np.min(image)
    residual[residual == 0] = np.min(image)
    
    if display:
        fig, ax = plt.subplots(1, 5, figsize=(20, 15))

        ax[0].set_title('HU Image')
        ax[0].imshow(image, cmap='gray')
        ax[0].axis('off')

        ax[1].set_title('Thresholded Image')
        ax[1].imshow(thresh_mask, cmap='gray')
        ax[1].axis('off')

        ax[2].set_title('Lungs Mask')
        ax[2].imshow(lungs_mask)
        ax[2].axis('off')

        ax[3].set_title('Lungs Image')
        ax[3].imshow(lungs)
        ax[3].axis('off')
        
        ax[4].set_title('Residual Image')
        ax[4].imshow(residual)
        ax[4].axis('off')
    
    return lungs, residual, lungs_mask

params = {
    'base_threshold': -700,
    'min_vperimeter': 2, 'min_hperimeter': 2,
    'min_varea': 6, 'min_harea': 6,
    'min_quantile_contours': 7,
    'perimeter_quantile': 0.1,
    'area_quantile': 0.1, 'area_max_quantile': 0.95,
    'min_vdistance': 30.0, 'min_hdistance': 3.0,
    'min_hv_ratio': 7.5
}

# '''
#     fixed -- ID00026637202179561894768 №11 -- strange images
#     fixed -- ID00076637202199015035026 №30 -- first 9 images need to be fixed
#     fixed -- ID00078637202199415319443 №32 -- two breathes
#     fixed -- ID00123637202217151272140 №51 -- very bag segmentation
#     fixed -- ID00126637202218610655908 №54 -- first 1 images need to be fixed
#     fixed -- ID00129637202219868188000 №57 -- very bag segmentation
#     fixed -- ID00128637202219474716089 №56 -- strange images
#     fixed -- 60 - strange image
#     fixed -- 63 - [309 - 336] bad segmentation
#     fixed -- 64 - [2-97] bad segmentation
#     fixed -- 67 - [51 -55], [69-70], [83] stuck together
#     fixed -- 68 - [187] !!no lungs!!
#     fixed -- 78 - [0 - 5], [42 - 55] bad segmentation
#     fixed -- 80 - [0-2] bad segmentation
#     fixed -- 81 - [295-307] stuck together (слиплись, но не очень сильно)
#     fixed -- 84 - [67-68] !!no lung!!
#     fixed -- 86 - [380, 386] - stuck together
#     fixed -- 96 - [23 -53] !!!no lungs!!!!
#     fixed -- 108 - from 158 bad segmentation
#     fixed -- 115 - [0-1] bad segmentation
#     fixed -- 121 -- bad (slices 149-157), but ok
# '''


