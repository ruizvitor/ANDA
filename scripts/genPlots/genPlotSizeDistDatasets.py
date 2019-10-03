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
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from scipy.stats import gaussian_kde

savePath="plots/"

def autolabel(rects, counts):
    # attach some text labels
    for ii, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, f"{counts[ii]:.2f}",
                 ha='center', va='bottom')

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def main():
    # fileList = os.listdir(FOLDER_IMG)

    # MEASURE_FOLDER = "/mnt/hd-data/bakrinski/KittiSegRUNS/fcn8_resnet50_fold0_testes_nvidia/testes/MSRA10Koriginal/masks/"
    # MEASURE_FOLDER = "/mnt/hd-data/bakrinski/KittiSegRUNS/train05/ResNet50msra10k/t00_fcn8_resnet50_all_2019_05_06_20.53/testes/MSRA10K/masks/"
    # MEASURE_FOLDER = "/mnt/hd-data/bakrinski/KittiSegRUNS/fcn8_resnet50_fold0_testes_nvidia/testes/MSRA10Koriginal/masks/"
    # MEASURE_FOLDER = "/mnt/hd-data/bakrinski/KittiSegRUNS/train05/Vgg16msra10k/t01_fcn8_vgg_fc7_all_2019_05_16_18.12/testes/MSRA10K/masks/"

    # n_bins=256
    # n_bins=100
    n_bins = 10
    # n_bins=1
    #DATASETS=["DUTOMRON","ECSSD","HKUIS","ICOSEG","PASCALS","SED1","SED2","THUR","MSRA10K"]
    #DATASETS = ["MSRA10Knew"]
    #DATASETS_NAME=["DUT-OMRON","ECSSD","HKU-IS","ICOSEG","PASCAL-S","SED1","SED2","THUR","MSRA10K"]


    # DATASETS=["DUTOMRON"]#,"ECSSD","HKUIS","ICOSEG","PASCALS","SED1","SED2","THUR","MSRA10K"]
    # DATASETS=["exp"]
    # DATASETS=["exp_filtred_17_05_2019_17h_with_MSRA10K"]
    # DATASETS=["exp_20_05_2019_with_MSRA10K"]
    # DATASETS=["Augmented MSRA10K 25_05"]
    DATASETS=["Augmented MSRA10K Experiment VIII"]
    DATASETS_NAME=["Augmented MSRA10K Experiment VIII"]
    j = 0
    for dataset in DATASETS:
        #FOLDER_MASK = "/home/bakrinski/datasets/"+dataset+"/masks/"
        # FOLDER_MASK = "/home/bakrinski/nobackup/datasets/" + dataset + "/masks/"
        # FOLDER_MASK = "multipleBG/masks/"
        FOLDER_MASK = "/home/dvruiz/scriptPosProcessObjects/29_05_2019_FullMix/multipleBG/masks/"
        # FOLDER_MASK = "filtered_17_05_2019_17h_with_MSRA10K/multipleBG/masks/"

        fileList = os.listdir(FOLDER_MASK)
        fileList = sorted(fileList)
        ys = np.zeros(len(fileList), np.float32)
        zs = np.zeros(len(fileList), np.float32)
        # ys=np.empty(len(fileList),np.float32)
        index = 0
        for i in fileList:

            maskName = i
            maskFile = Image.open(FOLDER_MASK + maskName)
            mask = np.array(maskFile)

            shape = mask.shape
            h = shape[0]
            w = shape[1]
            maskFile.close()

            ymin, ymax, xmin, xmax = bbox(mask)

            propX = (xmax - xmin)
            propY = (ymax - ymin)

            areaOBJ = propX * propY
            areaIMG = h * w

            prop = areaOBJ / areaIMG


            ys[index] = prop

            index += 1


        plt.clf()
        plt.title(DATASETS_NAME[j]+"\n Distribution of Bounding Boxes Size")

        weights = np.ones_like(ys) / float(len(ys))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        array_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

        counts, bins, patches = ax.hist(
            ys, weights=weights, bins=array_bins, zorder=10, label='n-samples')


        print("bins=", bins)

        farray = np.zeros(10)
        sarray = np.zeros(10)
        elem = np.zeros(10)
        inds = np.digitize(ys, bins[:len(bins) - 1])

        for i in range(0, len(zs)):
            farray[inds[i] - 1] += zs[i]
            sarray[inds[i] - 1] += 1
            elem[inds[i] - 1] += 1

        for i in range(0, len(farray)):
            if(elem[i] != 0):
                farray[i] /= elem[i]
            sarray[i] /= 10000

        print("farray=", farray)
        print("sarray=", sarray)

        print("counts.shape=", counts.shape)
        print("counts=", counts)

        autolabel(patches, counts)

        ax.set_title(DATASETS_NAME[j]+"\n Distribution of Bounding Boxes Size",fontsize="xx-large")
        ax.set_xlabel("Bounding Box Area Proportion",fontsize="xx-large")
        ax.set_xlim(0, 1)

        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylabel("Normalized Number of Samples",fontsize="xx-large")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid()

        plt.tight_layout()
        plt.savefig(savePath + dataset + 'size.png')
        plt.savefig(savePath + dataset + 'size.svg')
        plt.savefig(savePath + dataset + 'size.pdf')
        plt.savefig(savePath + dataset + 'size.eps')
        j+=1

if __name__ == '__main__':
    main()
