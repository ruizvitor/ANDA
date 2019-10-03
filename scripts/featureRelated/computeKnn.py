###############################################################################
#MIT License
#
#Copyright (c) 2019 Daniel Vitor Ruiz
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###############################################################################

import os
import cv2
import numpy as np
from PIL import Image
import time
import scipy
from sklearn.neighbors import NearestNeighbors
import skimage.feature as ft



#for fancy parameterization
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Compute feature vectors for the objects and backgrounds for the MSRA10K dataset')

    parser.add_argument(
        '-obj_path', '--obj_path',
        type=str, default="/home/bakrinski/datasets/MSRA10K/images/",
        help='OBJ_FOLDER_IMG input images path'
    )

    parser.add_argument(
        '-obj_mask_path', '--obj_mask_path',
        type=str, default="/home/bakrinski/datasets/MSRA10K/masks/",
        help='OBJ_FOLDER_MASK input masks path'
    )

    parser.add_argument(
        '-bg_path', '--bg_path',
        type=str, default="/home/dvruiz/PConv-Keras/output/",
        help='BG_FOLDER_IMG background images path'
    )

    parser.add_argument(
        '-metric_knn', '--metric_knn',
        type=str, default="cosine",
        help='distance function used in the knn'
    )

    parser.add_argument(
        '-nbins', '--nbins',
        type=int, default=64,
        help='number of bins for each histogram channel'
    )

    parser.add_argument(
        '-size', '--size',
        type=int, default=10000,
        help='number of images in the dataset'
    )

    parser.add_argument(
        '-k', '--k',
        type=int, default=10000,
        help='number of k NearestNeighbors'
    )

    return  parser.parse_args()


def getHistograms(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(imgHsv)

    histH, _ = np.histogram(h, bins=NBINS, density=True)
    histS, _ = np.histogram(s, bins=NBINS, density=True)
    histV, _ = np.histogram(v, bins=NBINS, density=True)

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # settings for LBP
    # radius = 3
    # n_points = 8 * radiusDATASET_SIZE
    # METHOD = 'uniform'
    # lbp = ft.local_binary_pattern(imgGray, n_points, radius, METHOD)
    lbp = ft.local_binary_pattern(imgGray, 24, 3, 'uniform')

    histLBP, _ = np.histogram(lbp, bins=NBINS, density=True)

    hist = np.concatenate((histH, histS, histV, histLBP))

    return hist


def getHistogramsWithMask(img, mask):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(imgHsv)

    histH, _ = np.histogram(h, bins=NBINS, density=True, weights=mask)
    histS, _ = np.histogram(s, bins=NBINS, density=True, weights=mask)
    histV, _ = np.histogram(v, bins=NBINS, density=True, weights=mask)

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # settings for LBP
    # radius = 3
    # n_points = 8 * radius
    # METHOD = 'uniform'
    # lbp = ft.local_binary_pattern(imgGray, n_points, radius, METHOD)
    lbp = ft.local_binary_pattern(imgGray, 24, 3, 'uniform')

    histLBP, _ = np.histogram(lbp, bins=NBINS, density=True, weights=mask)

    hist = np.concatenate((histH, histS, histV, histLBP))

    return hist

def main():

    #CALL PARSER
    args = parse_args()
    #

    # SETTINGS
    OBJ_FOLDER_IMG = args.obj_path
    OBJ_FOLDER_MASK = args.obj_mask_path
    BG_FOLDER_IMG = args.bg_path
    NBINS = args.nbins
    DATASET_SIZE = args.size
    N_NEIGHBORS = args.k
    METRIC_KNN = args.metric_knn
    ##

    existsDataSetFile = os.path.isfile('dataset.txt')
    if not(existsObj):
        with open('dataset.txt', 'w') as fd:
            for i in range(0,10000):
                print(i, file=fd)

    print("now obj")
    existsObj = os.path.isfile('histogramsOBJ.npy')
    if not(existsObj):
        print("building histograms")

        histogramsOBJ = np.empty((DATASET_SIZE, NBINS * 4), np.float32)
        for i in range(0, DATASET_SIZE, 1):
            imgName = "MSRA10K_image_{:06d}.jpg".format(i)
            imFile = Image.open(OBJ_FOLDER_IMG + imgName)
            img = np.array(imFile)

            maskName = imgName.replace(".jpg", ".png")
            maskName = maskName.replace("image", "mask")

            maskFile = Image.open(OBJ_FOLDER_MASK + maskName)
            mask = np.array(maskFile) / 255

            hist = getHistogramsWithMask(img, mask)

            histogramsOBJ[i] = hist

            imFile.close()
            maskFile.close()

        print("saving array histogramsOBJ.npy")
        np.save("histogramsOBJ.npy", histogramsOBJ)
    else:
        print("loading array histogramsOBJ.npy")
        histogramsOBJ = np.load("histogramsOBJ.npy")

    print("now bg")
    existsBg = os.path.isfile('histogramsBG.npy')
    if not(existsBg):
        print("building histograms")
        histogramsBG = np.empty((DATASET_SIZE, NBINS * 4), np.float32)
        for i in range(0, DATASET_SIZE):
            bgName = "MSRA10K_image_{:06d}.png".format(i)
            bgFile = Image.open(BG_FOLDER_IMG + bgName)
            bg = np.array(bgFile)

            hist = getHistograms(bg)

            histogramsBG[i] = hist

            bgFile.close()

        print("saving array histogramsBG.npy")
        np.save("histogramsBG.npy", histogramsBG)
    else:
        print("loading array histogramsBG.npy")
        histogramsBG = np.load("histogramsBG.npy")

    print("now knn")
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric=METRIC_KNN,
                            algorithm='auto', n_jobs=-1).fit(histogramsBG)

    distances, indices = nbrs.kneighbors(histogramsOBJ)

    HALF_N_NEIGHBORS= int(np.floor(N_NEIGHBORS/2))

    with open('distances_' + METRIC_KNN + '.txt', 'w') as fd:
        with open('indices_' + METRIC_KNN + '.txt', 'w') as fi:
            for i in range(0, N_NEIGHBORS):
                valuesDis = str(distances[i][HALF_N_NEIGHBORS])
                valuesIndex = str(indices[i][HALF_N_NEIGHBORS])

                print(valuesDis, file=fd)
                print(valuesIndex, file=fi)


main()
