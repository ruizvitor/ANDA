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
from scipy.stats import gaussian_kde

savePath="plots/"

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def main():
    # fileList = os.listdir(FOLDER_IMG)

    #DATASETS=["DUTOMRON","ECSSD","HKUIS","ICOSEG","PASCALS","SED1","SED2","THUR","MSRA10K"]
    #DATASETS_NAME=["DUT-OMRON","ECSSD","HKU-IS","ICOSEG","PASCAL-S","SED1","SED2","THUR","MSRA10K"]

    #DATASETS=["DUTOMRON","ECSSD","HKUIS","ICOSEG","PASCALS","SED1","SED2","THUR","MSRA10K"]
    #DATASETS_NAME=["DUT-OMRON","ECSSD","HKU-IS","ICOSEG","PASCAL-S","SED1","SED2","THUR","MSRA10K"]

    # DATASETS=["exp"]
    # DATASETS=["exp_filtred_17_05_2019_17h_with_MSRA10K"]
    # DATASETS=["exp_20_05_2019_with_MSRA10K"]
    # DATASETS=["Augmented MSRA10K 29_05"]
    DATASETS=["Augmented MSRA10K Experiment VIII"]
    DATASETS_NAME=["Augmented MSRA10K Experiment VIII"]
    j = 0
    for dataset in DATASETS:
        #FOLDER_MASK = "/home/bakrinski/datasets/"+dataset+"/masks/"
        # FOLDER_MASK = "multipleBG/masks/"
        FOLDER_MASK = "/home/dvruiz/scriptPosProcessObjects/29_05_2019_FullMix/multipleBG/masks/"
        # FOLDER_MASK = "tmp/masks/"
        # FOLDER_MASK = "filtered_17_05_2019_17h_with_MSRA10K/multipleBG/masks/"
        # FOLDER_MASK = "20_05_2019_with_MSRA10K/multipleBG/masks/"


        fileList = os.listdir(FOLDER_MASK)
        xs=np.empty(len(fileList),np.float32)
        ys=np.empty(len(fileList),np.float32)
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
            centerx = ((xmax-xmin)/2)+xmin
            centery = ((ymax-ymin)/2)+ymin

            newx = centerx/w
            newy = centery/h
            xs[index]=newx
            ys[index]=newy
            index+=1

        plt.clf()
        plt.title(DATASETS_NAME[j]+"\n Distribution of Bounding Boxes Center Coordinates",fontsize="xx-large")
        plt.xlabel("Normalized Position X",fontsize="xx-large")
        plt.xlim(0, 1)
        plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.ylabel("Normalized Position Y",fontsize="xx-large")
        plt.ylim(0, 1)
        plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

        xy = np.vstack([xs,ys])
        z = gaussian_kde(xy)(xy)
        z = z/100

        plt.scatter(xs, ys, c=z, s=10, edgecolor='', vmin=0, vmax=0.5, cmap=plt.get_cmap('hot'))

        cb = plt.colorbar()
        cb.set_label("Sample Density",fontsize="xx-large")

        plt.tight_layout()
        plt.savefig(savePath+dataset+'pos.png')
        plt.savefig(savePath+dataset+'pos.pdf')
        plt.savefig(savePath+dataset+'pos.svg')
        plt.savefig(savePath+dataset+'pos.eps')
        j+=1



if __name__ == '__main__':
    main()
