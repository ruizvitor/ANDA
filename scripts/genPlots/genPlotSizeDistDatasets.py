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
# from scipy.interpolate import interpn

savePath="plots/"

def autolabel(rects, counts):
    # attach some text labels
    for ii, rect in enumerate(rects):
        height = rect.get_height()
        # f"{i:.2f}"
        # plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, counts[ii],
        #         ha='center', va='bottom')
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

            # fmeaName = i.replace(".png", ".txt")
            # fmeaName = fmeaName.replace("mask", "image")
            #
            # f = open(MEASURE_FOLDER + fmeaName, "r")
            # line = f.readline()
            # line = f.readline()
            # line = f.readline()
            # print(line)
            # args = line.split(":")
            # fmeasure = float(args[1])
            # zs[index] = fmeasure
            # f.close()

            # print(i)
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

            propX = (xmax - xmin)
            propY = (ymax - ymin)

            areaOBJ = propX * propY
            areaIMG = h * w

            prop = areaOBJ / areaIMG
            # print(prop)

            ys[index] = prop

            index += 1

        # print(dataset+"=",ys)
        #
        # with open(dataset+'_sizes'+'.txt', 'w') as f:
        #     for i in ys:
        #         print(i, file=f)

        plt.clf()
        plt.title(DATASETS_NAME[j]+"\n Distribution of Bounding Boxes Size")

        weights = np.ones_like(ys) / float(len(ys))
        # np.hist()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        array_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

        counts, bins, patches = ax.hist(
            ys, weights=weights, bins=array_bins, zorder=10, label='n-samples')

        # minbin = 0.
        # maxbin = 1.
        # bins = np.linspace(minbin,maxbin,n_bins)
        #
        # cmap = plt.cm.jet
        # norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        # colors = cmap(bins)
        #
        # fig = plt.figure()
        # ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.1])
        # cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
        #                                 norm=norm,
        #                                 orientation='horizontal')

        # counts,bins,patches = plt.hist2d(ys, zs, weights=weights, bins=n_bins)
        # plt.hist2d(ys, zs, weights=weights, bins=n_bins)
        # cb = plt.colorbar()

        # We'll color code by height, but you could use any scalar
        # fracs = counts / counts.max()

        print("bins=", bins)

        farray = np.zeros(10)
        sarray = np.zeros(10)
        elem = np.zeros(10)
        # array_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        inds = np.digitize(ys, bins[:len(bins) - 1])

        for i in range(0, len(zs)):
            farray[inds[i] - 1] += zs[i]
            sarray[inds[i] - 1] += 1
            elem[inds[i] - 1] += 1

        for i in range(0, len(farray)):
            if(elem[i] != 0):
                farray[i] /= elem[i]
            sarray[i] /= 10000

            # print("farray[",i,"]=",farray[i])
            # print("sarray[",i,"]=",sarray[i])

        print("farray=", farray)
        print("sarray=", sarray)

        print("counts.shape=", counts.shape)
        print("counts=", counts)

        # ax2 = ax.twinx()
        # shifted_x = bins[:len(bins) - 1]
        # shifted_x+=0.05
        # plt.plot(shifted_x, farray, '-r', marker='o', label='f-measure')
        # ax2.plot(time, temp, '-r', label = 'temp')


        # plt.bar(bins[:len(bins)-1],farray)

        # we need to normalize the data to 0..1 for the full range of the colormap
        # norm = colors.Normalize(farray.min(), farray.max())
        #
        # # Now, we'll loop through our objects and set the color of each accordingly
        # for thisfrac, thispatch in zip(farray, patches):
        #     color = plt.cm.jet(norm(thisfrac))
        #     thispatch.set_facecolor(color)

        # cm = plt.cm.RdBu_r
        #
        # # n, bins, patches = plt.hist(data, 25, normed=1, color='green')
        # for i, p in enumerate(patches):
        #     plt.setp(p, 'facecolor', cm(i/10)) # notice the i/25

        # bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        # for count, x in zip(counts, bin_centers):
        #     # Label the raw counts
        #     plt.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
        #         xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # plt.colorbar()

        autolabel(patches, counts)

        # counts,bins,patches = plt.hist(farray, bins=n_bins)
        # bar = plt.bar(bins[:len(bins)-1],farray,width=0.1, color="red", zorder=0)
        # bar.set_zorder(0)  # put the legend on top

        # fig.title(dataset + "\n Distribution of Bounding Boxes Size")
        #
        ax.set_title(DATASETS_NAME[j]+"\n Distribution of Bounding Boxes Size",fontsize="xx-large")
        ax.set_xlabel("Bounding Box Area Proportion",fontsize="xx-large")
        ax.set_xlim(0, 1)
        # ax.xlim(0, 1)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylabel("Normalized Number of Samples",fontsize="xx-large")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid()
        # ax.legend(loc=2)
        #
        # ax2.set_ylabel("Mean F-measure per Bin")
        # ax2.set_ylim(0, 1)
        # ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # plt.legend(loc=2)
        # h1, l1 = ax.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        # ax.legend(h1+h2, l1+l2, loc=2)

        plt.tight_layout()
        plt.savefig(savePath + dataset + 'size.png')
        plt.savefig(savePath + dataset + 'size.svg')
        plt.savefig(savePath + dataset + 'size.pdf')
        plt.savefig(savePath + dataset + 'size.eps')
        j+=1

if __name__ == '__main__':
    main()
