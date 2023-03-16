from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix
from _collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import SimpleITK as sitk
import cv2

def image_resample(img, ref=None, spacing_new=(1.0, 1.0, 5.0), is_label=False):
    '''

    :param img:
    :param ref: reference image
    :param spacing_new:  x,y,z, spacing for output, if ref is available, this param will be ignored
    :param is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    :return: output image

    '''

    resample = sitk.ResampleImageFilter()

    if ref:
        resample.SetReferenceImage(ref)
    else:
        size = np.array(img.GetSize())
        spacing = np.array(img.GetSpacing())

        spacing_new = np.array(spacing_new)
        size_new = [int(s) for s in (size * spacing / spacing_new)]

        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputOrigin(img.GetOrigin())
        resample.SetOutputSpacing(spacing_new)
        resample.SetSize(size_new)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(img)
    return newimage

def MedImgKnn(img, label, k):
    sitk.ProcessObject_SetGlobalDefaultCoordinateTolerance(1e-3) #TODO
    sitk.ProcessObject_SetGlobalDefaultDirectionTolerance(1e-3)

    label = sitk.Cast(label, sitk.sitkUInt16)
    img = sitk.Cast(img, sitk.sitkUInt16)

    so_roi = sitk.Multiply(img, label)
    arr_roi = sitk.GetArrayFromImage(so_roi)

    rows, cols, height = arr_roi.shape[:]
    arr1d = arr_roi.reshape((rows * cols * height, 1))
    arr1d = np.float32(arr1d)

    loc_valid = np.where(arr1d)
    arr1d_valid = arr1d[arr1d>0]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, arr_subroi_valid, centers = cv2.kmeans(arr1d_valid, k, None, criteria, 10, flags)

    for i, ind_roi in enumerate(arr_subroi_valid):
        arr1d[loc_valid[0][i], 0] = arr_subroi_valid[i][0] + 1

    arr_subroi = arr1d.reshape((rows, cols, height))
    arr_subroi = arr_subroi.astype(np.int16)
    so_subroi = sitk.GetImageFromArray(arr_subroi)
    so_subroi.CopyInformation(img)

    return so_subroi


def read_so_series(path, ind=0):
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_IDs[ind])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    so_img = series_reader.Execute()
    return so_img
