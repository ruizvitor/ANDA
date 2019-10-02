from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet
import os
import cv2
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

#for fancy parameterization
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Simple script to directly use PConv inpainting for the MSRA10K dataset')

    parser.add_argument(
        '-img_path', '--img_path',
        type=str, default="/home/bakrinski/datasets/MSRA10K/images/",
        help='TEST_FOLDER_IMG input images path'
    )

    parser.add_argument(
        '-mask_path', '--mask_path',
        type=str, default="/home/bakrinski/datasets/MSRA10K/masks/",
        help='TEST_FOLDER_MASK input masks path'
    )

    parser.add_argument(
        '-out_path', '--out_path',
        type=str, default="output/",
        help='OUTPUT_FOLDER path'
    )

    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=4,
        help='What batch-size should we use'
    )

    return  parser.parse_args()

def main():

    #CALL PARSER
    args = parse_args()
    #

    # Change to root path
    if os.path.basename(os.getcwd()) != 'nvidiaInpainting':
        os.chdir('..')


    # SETTINGS
    TEST_FOLDER_IMG = args.img_path
    TEST_FOLDER_MASK = args.mask_path
    OUTPUT_FOLDER = args.out_path
    BATCH_SIZE = args.batch_size
    #

    model = PConvUnet(vgg_weights=None, inference_only=True)
    model.load("pconv_imagenet.h5", train_bn=False)

    fileList = os.listdir(TEST_FOLDER_IMG)

    # Used for chunking up images & stiching them back together
    chunker = ImageChunker(512, 512, 30)
    kernel = np.ones((7, 7), np.uint8)

    for i in range(0, len(fileList), BATCH_SIZE):
        ####
        # Lists for saving images and masks
        imgs, masks, indices = [], [], []
        for j in range(0, BATCH_SIZE):
            imgName = "MSRA10K_image_{:06d}.jpg".format(i + j)

            imFile = Image.open(TEST_FOLDER_IMG + imgName)
            im = np.array(imFile) / 255  # convert to float


            maskName = imgName.replace(".jpg", ".png")
            maskName = maskName.replace("image", "mask")

            maskFile = Image.open(TEST_FOLDER_MASK + maskName)
            mask = np.array(maskFile)

            # extend from 1 channel to 3
            mask3d = np.tile(mask[:, :, None], [1, 1, 3])

            # dilate mask to process additional border
            mask3d = cv2.dilate(mask3d, kernel, iterations=1)
            mask3d = mask3d / 255  # convert to float
            mask3d = 1.0 - mask3d  # need to invert mask due to framework

            imgs.append(im)
            masks.append(mask3d)
            indices.append(i+j)

            imFile.close()
            maskFile.close()
            print(imgName, maskName)
            ####

        # print("testing....")
        for img, mask, index in zip(imgs, masks, indices):

            ###begin resize

            height, width, depth = img.shape
            imgScale = 0.5
            newX,newY = int(width*imgScale),int(height*imgScale)


            new_img = cv2.resize(img,(newX,newY))
            new_mask = cv2.resize(mask,(newX,newY))

            chunked_images = chunker.dimension_preprocess(deepcopy(new_img))
            chunked_masks = chunker.dimension_preprocess(deepcopy(new_mask))
            pred_imgs = model.predict([chunked_images, chunked_masks])

            reconstructed_image_resized = chunker.dimension_postprocess(pred_imgs, new_img)
            reconstructed_image_original_size = cv2.resize(reconstructed_image_resized,(int(width),int(height)))


            maskExpanded = cv2.erode(mask, kernel, iterations=3)

            reconstructed_image_final = np.where(maskExpanded == 0, reconstructed_image_original_size, img)#apply generated over masked area only

            result = Image.fromarray((reconstructed_image_final * 255).astype(np.uint8))
            result.save(OUTPUT_FOLDER+"MSRA10K_image_{:06d}.png".format(index))

main()
