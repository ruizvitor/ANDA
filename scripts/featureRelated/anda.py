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

#for fancy parameterization
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Compute resulting image using ANDA techinique for the MSRA10K dataset')

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
        '-index_obj_path', '--index_obj_path',
        type=str, default="dataset.txt",
        help='LIST_OF_N_OBJECTS filepath for the file containing per line a indice, e.g. "dataset.txt" resulting from genObjIndicees.py'
    )

    parser.add_argument(
        '-index_bg_path', '--index_bg_path',
        type=str, default="indices_cosine.txt",
        help='LIST_OF_INDICES filepath for the file containing per line a indice, e.g. "indices_cosine.txt" resulting from computeKnn.py'
    )

    parser.add_argument(
        '-out_path', '--out_path',
        type=str, default="output/",
        help='output path containing a folder named images and masks, e.g."output/" '
    )

    parser.add_argument(
        '-seed', '--seed',
        type=int, default=22,
        help='seed number for the pseudo-random computation'
    )

    parser.add_argument(
        '-size', '--size',
        type=int, default=10000,
        help='number of images in the dataset'
    )


    parser.add_argument(
        '-n_bgs', '--n_bgs',
        type=int, default=1,
        help='N_OF_BACKGROUNDS'
    )


    parser.add_argument(
        '-n_ops', '--n_ops',
        type=int, default=1,
        help='N_OF_OPS'
    )

    return  parser.parse_args()

# SETTINGS

#CALL PARSER
args = parse_args()
#

OBJ_FOLDER_IMG = args.obj_path
OBJ_FOLDER_MASK = args.obj_mask_path
BG_FOLDER_IMG = args.bg_path
OUTPUT_FOLDER_IMG = "images/"
OUTPUT_FOLDER_MASK = "masks/"
LIST_OF_N_OBJECTS = args.index_obj_path
N_OBJECT = args.size
N_OF_BACKGROUNDS = args.n_bgs
N_OF_OPS = args.n_ops
LIST_OF_INDICES = args.index_bg_path

kernelErode = np.ones((3, 3), np.uint8)

maxH = 512
maxW = 512

random.seed(args.seed)
np.random.seed(args.seed)
noise_scale = np.random.uniform(low=0.975, high=1.025, size=N_OBJECT)
#

# # SETTINGS
# OBJ_FOLDER_IMG = "/home/bakrinski/datasets/MSRA10K/images/"
# OBJ_FOLDER_MASK = "/home/bakrinski/datasets/MSRA10K/masks/"
# BG_FOLDER_IMG = "/home/dvruiz/PConv-Keras/output/"
# OUTPUT_FOLDER_IMG = "images/"
# OUTPUT_FOLDER_MASK = "masks/"
# LIST_OF_N_OBJECTS = "dataset.txt"
# N_WORST = 10000
# N_OF_BACKGROUNDS = 1
# N_OF_OPS = 1
# LIST_OF_INDICES = "indices_cosine.txt"
#
# kernelErode = np.ones((3, 3), np.uint8)
#
# maxH = 512
# maxW = 512
#
# random.seed(22)
# np.random.seed(22)
# # noise_scale = np.random.uniform(low=0.975, high=1.025, size=13980)
# noise_scale = np.random.uniform(low=0.975, high=1.025, size=N_WORST)
# #

def randomTranslateInside(newYmax, newYmin, newXmax, newXmin, newOrigin, border, M):
    noise_x = np.random.uniform(low=0.0, high=1.0)
    noise_y = np.random.uniform(low=0.0, high=1.0)
    # check if bbox can move in y
    if((newYmax - newYmin) < border[0]):
        # check the direction of free space
        if((newYmax) < newOrigin[0] + border[0]):
            if((newYmin) > newOrigin[0]):
                freeSpacePos = (newOrigin[0] + border[0]) - newYmax
                freeSpaceNeg = newYmin - newOrigin[0]

                luck = np.random.randint(low=0, high=2)
                if(luck == 0):
                    M[1][2] += np.floor(noise_y * freeSpacePos)
                else:
                    M[1][2] -= np.floor(noise_y * freeSpaceNeg)

            else:
                freeSpace = (newOrigin[0] + border[0]) - newYmax
                M[1][2] +=  np.floor(noise_y * freeSpace)
        else:
            if((newYmin) > newOrigin[0]):
                freeSpace = newYmin - newOrigin[0]
                M[1][2] -=  np.floor(noise_y * freeSpace)

    if((newXmax - newXmin) < border[1]):
        # check the direction of free space
        if((newXmax) < newOrigin[1] + border[1]):
            if((newXmin) > newOrigin[1]):
                freeSpacePos = (newOrigin[1] + border[1]) - newXmax
                freeSpaceNeg = newXmin - newOrigin[1]

                luck = np.random.randint(low=0, high=2)
                if(luck == 0):
                    M[0][2] += np.floor(noise_x * freeSpacePos)
                else:
                    M[0][2] -= np.floor(noise_x * freeSpaceNeg)

            else:
                freeSpace = (newOrigin[1] + border[1]) - newXmax
                M[0][2] +=  np.floor(noise_x * freeSpace)
        else:
            if((newXmin) > newOrigin[1]):
                freeSpace = newXmin - newOrigin[1]
                M[0][2] -=  np.floor(noise_x * freeSpace)
    return M


def geometricOp2(resizedImg, resizedMask, bgOriginalshape, op, globalIndex):
    #######################################################
    diffH = int((resizedImg.shape[0] - bgOriginalshape[0]) / 2)
    diffW = int((resizedImg.shape[1] - bgOriginalshape[1]) / 2)
    ####
    ymin, ymax, xmin, xmax = bbox(resizedMask)

    # xmin -= np.abs(noise_translate_x[globalIndex])
    # xmax += np.abs(noise_translate_x[globalIndex])
    # ymin -= np.abs(noise_translate_y[globalIndex])
    # ymax += np.abs(noise_translate_y[globalIndex])

    propX = (xmax - xmin)
    propY = (ymax - ymin)

    areaOBJ = propX * propY
    areaIMG = bgOriginalshape[0] * bgOriginalshape[1]

    prop = areaOBJ / areaIMG

    ###

    op = globalIndex % 5

    if(op == 0):
        beta = 0.05 * noise_scale[globalIndex]
    if(op == 1):
        beta = 0.15 * noise_scale[globalIndex]
    if(op == 2):
        beta = 0.65 * noise_scale[globalIndex]
    if(op == 3):
        beta = 0.75 * noise_scale[globalIndex]
    if(op == 4):
        beta = 0.85 * noise_scale[globalIndex]

    scale = np.sqrt((beta * areaIMG) / areaOBJ)

    diffx = ((xmax - xmin) / 2)
    diffy = ((ymax - ymin) / 2)
    centerx = xmin + diffx
    centery = ymin + diffy

    pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax]])

    newXmin = centerx - diffx * scale
    newXmax = centerx + diffx * scale

    newYmin = centery - diffy * scale
    newYmax = centery + diffy * scale

    # LOGIC HERE
    newOrigin = [diffH, diffW]
    border = [bgOriginalshape[0], bgOriginalshape[1]]

    # check if the aspect of the object is the same as the bg
    obj_orientation = -1
    bg_orientation = -1

    if(diffx >= diffy):
        obj_orientation = 0
    else:
        obj_orientation = 1

    if(bgOriginalshape[1] >= bgOriginalshape[0]):
        bg_orientation = 0
    else:
        bg_orientation = 1

    # check if can fit
    if((newYmax - newYmin <= border[0])and(newXmax - newXmin <= border[1])):
        # ok then it can fit
        # but does it need translation?

        pts2 = np.float32(
            [[newXmin, newYmin], [newXmax, newYmin], [newXmin, newYmax]])

        M = cv2.getAffineTransform(pts1, pts2)

        # origin of object must be >= newOrigin
        if(newYmin <= newOrigin[0]):
            local_diff_y = newOrigin[0] - newYmin
            M[1][2] += (local_diff_y)

        if(newXmin <= newOrigin[1]):
            local_diff_x = newOrigin[1] - newXmin
            M[0][2] += (local_diff_x)

        # maxdim must be <= border with the correct origin
        if(newYmax >= (border[0] + newOrigin[0])):
            local_diff_y = newYmax - (border[0] + newOrigin[0])
            M[1][2] -= (local_diff_y)

        if(newXmax >= (border[1] + newOrigin[1])):
            local_diff_x = newXmax - (border[1] + newOrigin[1])
            M[0][2] -= (local_diff_x)

        newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
        newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

        newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
        newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

        newXminTmp = min(newXmin, newXmax)
        newXmaxTmp = max(newXmin, newXmax)

        newYminTmp = min(newYmin, newYmax)
        newYmaxTmp = max(newYmin, newYmax)

        newXmin = newXminTmp
        newXmax = newXmaxTmp

        newYmin = newYminTmp
        newYmax = newYmaxTmp

        M = randomTranslateInside(
            newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)

        resizedImg = cv2.warpAffine(
            resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
        resizedMask = cv2.warpAffine(
            resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)
    else:
        # it cannot fit
        # resize
        if(obj_orientation == bg_orientation):
            # print("same")
            # limit resize to max that fits

            # scale must consider translation
            scale = min((border[0]) / (ymax - ymin),
                        (border[1]) / (xmax - xmin))
            #
            newXmin = centerx - diffx * scale
            newXmax = centerx + diffx * scale

            newYmin = centery - diffy * scale
            newYmax = centery + diffy * scale

            pts2 = np.float32(
                [[newXmin, newYmin], [newXmax, newYmin], [newXmin, newYmax]])

            M = cv2.getAffineTransform(pts1, pts2)

            # origin of object must be >= newOrigin
            if(newYmin <= newOrigin[0]):
                local_diff_y = newOrigin[0] - newYmin
                M[1][2] += (local_diff_y)

            if(newXmin <= newOrigin[1]):
                local_diff_x = newOrigin[1] - newXmin
                M[0][2] += (local_diff_x)

            # maxdim must be <= border with the correct origin
            if(newYmax >= (border[0] + newOrigin[0])):
                local_diff_y = newYmax - (border[0] + newOrigin[0])
                M[1][2] -= (local_diff_y)

            if(newXmax >= (border[1] + newOrigin[1])):
                local_diff_x = newXmax - (border[1] + newOrigin[1])
                M[0][2] -= (local_diff_x)

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp
            #
            M = randomTranslateInside(
                newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)

            resizedImg = cv2.warpAffine(
                resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
            resizedMask = cv2.warpAffine(
                resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)
        else:
            # print("different")
            # check if a rotated obj fits

            idxmod = np.random.randint(low=0, high=2)
            if(idxmod == 0):
                degrees = -90
            if(idxmod == 1):
                degrees = 90

            M = cv2.getRotationMatrix2D(((maxW / 2), (maxH / 2)), degrees, 1)

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp

            # scale must consider translation
            scale = min((border[0]) / (newYmax - newYmin),
                        (border[1]) / (newXmax - newXmin))
            #

            M[0][0] *= scale
            M[0][1] *= scale
            M[1][0] *= scale
            M[1][1] *= scale

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp

            # origin of object must be >= newOrigin
            if(newYmin <= newOrigin[0]):
                local_diff_y = newOrigin[0] - newYmin
                M[1][2] += (local_diff_y)

            if(newXmin <= newOrigin[1]):
                local_diff_x = newOrigin[1] - newXmin
                M[0][2] += (local_diff_x)

            # maxdim must be <= border with the correct origin
            if(newYmax >= (border[0] + newOrigin[0])):
                local_diff_y = newYmax - (border[0] + newOrigin[0])
                M[1][2] -= (local_diff_y)

            if(newXmax >= (border[1] + newOrigin[1])):
                local_diff_x = newXmax - (border[1] + newOrigin[1])
                M[0][2] -= (local_diff_x)

            newXmin = xmin * M[0][0] + ymin * M[0][1] + M[0][2]
            newXmax = xmax * M[0][0] + ymax * M[0][1] + M[0][2]

            newYmin = xmin * M[1][0] + ymin * M[1][1] + M[1][2]
            newYmax = xmax * M[1][0] + ymax * M[1][1] + M[1][2]

            newXminTmp = min(newXmin, newXmax)
            newXmaxTmp = max(newXmin, newXmax)

            newYminTmp = min(newYmin, newYmax)
            newYmaxTmp = max(newYmin, newYmax)

            newXmin = newXminTmp
            newXmax = newXmaxTmp

            newYmin = newYminTmp
            newYmax = newYmaxTmp
            #
            M = randomTranslateInside(
                newYmax, newYmin, newXmax, newXmin, newOrigin, border, M)

            resizedImg = cv2.warpAffine(
                resizedImg, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
            resizedMask = cv2.warpAffine(
                resizedMask, M, (maxW, maxH), flags=cv2.INTER_NEAREST)

    ####
    # cv2.rectangle(resizedMask, (int(newXmin), int(newYmin)),
    #               (int(newXmax), int(newYmax)), (255, 255, 255), 1)
    #######################################################
    return resizedImg, resizedMask

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def resize_with_pad(image, height, width):

    def get_padding_size(image, height, width):
        # h, w, _ = image.shape
        h = image.shape[0]
        w = image.shape[1]
        top, bottom, left, right = (0, 0, 0, 0)
        if h < height:
            dh = height - h
            top = dh // 2
            bottom = dh - top
        if w < width:
            dw = width - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image, height, width)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant


def resizeToOrg(bgOriginalshape, new, newMask):
    if(bgOriginalshape[0] < new.shape[0]):
        diffH = int((new.shape[0] - bgOriginalshape[0]) / 2)
        new = new[diffH:bgOriginalshape[0] + diffH, :, :]
        newMask = newMask[diffH:bgOriginalshape[0] + diffH, :, :]

    if(bgOriginalshape[1] < new.shape[1]):
        diffW = int((new.shape[1] - bgOriginalshape[1]) / 2)
        new = new[:, diffW:bgOriginalshape[1] + diffW, :]
        newMask = newMask[:, diffW:bgOriginalshape[1] + diffW, :]

    return new, newMask


def loadResizedBG(index):
    bgName = "MSRA10K_image_{:06d}.png".format(index)
    bgFile = Image.open(BG_FOLDER_IMG + bgName)
    bg = np.array(bgFile)
    bgOriginalshape = bg.shape
    resizedBg = resize_with_pad(bg, height=maxH, width=maxW)
    bgFile.close()
    return resizedBg, bgOriginalshape

def main(op, multipleBgs, outPath):

    # read LIST_OF_N_OBJECTS
    arrOBJ = np.zeros(N_OBJECT, np.int)
    f = open(LIST_OF_N_OBJECTS, "r")
    for i in range(0, N_OBJECT):
        line = f.readline()
        args = line.split(" ")
        arrOBJ[i] = int(args[0])
    f.close()
    ###

    # read LIST_OF_N_OBJECTS
    arrBG = np.zeros((N_OBJECT, N_OF_BACKGROUNDS), np.int)
    f = open(LIST_OF_INDICES, "r")
    for i in range(0, N_OBJECT):
        line = f.readline()
        if line == '\n':
            arrOBJ[i] = -1
        else:
            args = line.split(" ")
            for bgindex in range(0, N_OF_BACKGROUNDS):
                arrBG[i][bgindex] = int(args[bgindex])
    f.close()
    ###

    realI = 0

    for i in range(0, N_OBJECT, 1):
        if(arrOBJ[i] != -1):
            imgName = "MSRA10K_image_{:06d}.jpg".format(arrOBJ[i])
            imFile = Image.open(OBJ_FOLDER_IMG + imgName)
            img = np.array(imFile)

            maskName = imgName.replace(".jpg", ".png")
            maskName = maskName.replace("image", "mask")

            maskFile = Image.open(OBJ_FOLDER_MASK + maskName)

            mask = np.array(maskFile)

            mask = np.tile(mask[:, :, None], [1, 1, 3])

            resizedImg = resize_with_pad(img, height=maxH, width=maxW)
            resizedMask = resize_with_pad(mask, height=maxH, width=maxW)

            imFile.close()
            maskFile.close()
            # print(stamp)

            resizedImgArr = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            resizedMaskArr = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            # print(resizedImgArr)

            resizedBg = [None] * (N_OF_BACKGROUNDS)
            bgOriginalshape = [None] * (N_OF_BACKGROUNDS)

            blur = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            inv_blur = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            new = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            result = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS
            resizedMaskFinal = [[None] * (N_OF_OPS)] * N_OF_BACKGROUNDS

            for bgindex in range(0, N_OF_BACKGROUNDS):
                resizedBg[bgindex], bgOriginalshape[bgindex] = loadResizedBG(
                    arrBG[i][bgindex])

                # calcule ops per bgs
                for opindex in range(0, N_OF_OPS):
                    globalIndex = (
                        ((realI * N_OF_BACKGROUNDS) + bgindex) * N_OF_OPS) + opindex
                    # print(globalIndex)
                    resizedImgArr[bgindex][opindex], resizedMaskArr[bgindex][opindex] = geometricOp2(
                        resizedImg, resizedMask, bgOriginalshape[bgindex], opindex, globalIndex)

                    # internalLoop
                    # BEGIN Smooth border copy
                    resizedMaskTmp = cv2.erode(
                        resizedMaskArr[bgindex][opindex], kernelErode, iterations=1)
                    blur[bgindex][opindex] = cv2.blur(resizedMaskTmp, (3, 3))

                    blur[bgindex][opindex] = (
                        blur[bgindex][opindex] / 255) * 0.95
                    inv_blur[bgindex][opindex] = 1 - blur[bgindex][opindex]

                    new[bgindex][opindex] = blur[bgindex][opindex] * resizedImgArr[bgindex][opindex] + \
                        inv_blur[bgindex][opindex] * resizedBg[bgindex]
                    # END Smooth border copy

                    new[bgindex][opindex], resizedMaskArr[bgindex][opindex] = resizeToOrg(
                        bgOriginalshape[bgindex], new[bgindex][opindex], resizedMaskArr[bgindex][opindex])

                    #########################################################

                    result[bgindex][opindex] = Image.fromarray(
                        (new[bgindex][opindex]).astype(np.uint8))

                    resizedMaskFinal[bgindex][opindex] = Image.fromarray(
                        (resizedMaskArr[bgindex][opindex]).astype(np.uint8))

                    stamp = "{:06d}_{:06d}_{:03d}.png".format(
                        arrOBJ[i], arrBG[i][bgindex], opindex)

                    result[bgindex][opindex].save(outPath + OUTPUT_FOLDER_IMG +
                                                  "MSRA10K_image_" + stamp)
                    resizedMaskFinal[bgindex][opindex].save(outPath + OUTPUT_FOLDER_MASK
                                                            + "MSRA10K_mask_" + stamp)

                    print(stamp)
                    #########################################################

            realI += 1


if __name__ == '__main__':

    if(args.n_bgs>1):
        main(0,True,args.out_path)
    else:
        main(0,False,args.out_path)
