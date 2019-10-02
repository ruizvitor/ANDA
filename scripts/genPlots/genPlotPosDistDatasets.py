import os
import cv2
import numpy as np
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# from scipy.interpolate import interpn

# SETTINGS

# def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
#     """
#     Scatter plot colored by 2d histogram
#     """
#     if ax is None :
#         fig , ax = plt.subplots()
#     data , x_e, y_e = np.histogram2d( x, y, bins = bins)
#     z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )
#
#     # Sort the points by density, so that the densest points are plotted last
#     if sort :
#         idx = z.argsort()
#         x, y, z = x[idx], y[idx], z[idx]
#
#     ax.scatter( x, y, c=z, s=10,  **kwargs )
#     return ax

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
            # h,w = mask.shape
            # h,w,_ = mask.shape
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
